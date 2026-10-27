const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// Exercise the actual hook/effect boundary with deterministic browser time.
function browser({hidden=false, request=async () => ({ok:true})}={}) {
  let time=0, calls=0, nextId=0, state;
  const timers=new Map(), effects=[], listeners=new Set();
  const document={hidden, addEventListener: (_,fn)=>listeners.add(fn), removeEventListener: (_,fn)=>listeners.delete(fn)};
  const schedule=(fn,ms,repeat=0)=>{const id=++nextId;timers.set(id,{fn,at:time+ms,repeat});return id;};
  const context=vm.createContext({document,AbortController,Date:class extends Date {static now(){return time;}},
    setTimeout:(fn,ms)=>schedule(fn,ms),clearTimeout:id=>timers.delete(id),
    setInterval:(fn,ms)=>schedule(fn,ms,ms),clearInterval:id=>timers.delete(id),
    useState:initial=>{state=initial;return [initial,value=>{state=value;}];},useEffect:fn=>effects.push(fn),
    apiFetch:(...args)=>{calls++;return request(...args);}});
  const source=fs.readFileSync(path.join(__dirname,'useBackendHealth.js'),'utf8').replace(/^import .*$/gm,'').replace(/export /g,'');
  vm.runInContext(source,context);
  const flush=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
  return {
    get calls(){return calls;},get state(){return state;},
    async mount(){effects.length=0;vm.runInContext('useBackendHealth()',context);const cleanup=effects.map(fn=>fn());await flush();return ()=>cleanup.forEach(fn=>fn?.());},
    async visible(value){document.hidden=!value;listeners.forEach(fn=>fn());await flush();},
    async retry(){vm.runInContext('retryBackendHealth()',context);await flush();},
    async advance(ms){const target=time+ms;while(true){const next=[...timers.entries()].filter(([,t])=>t.at<=target).sort((a,b)=>a[1].at-b[1].at)[0];if(!next)break;const [id,t]=next;time=t.at;if(t.repeat)t.at+=t.repeat;else timers.delete(id);t.fn();await flush();}time=target;await flush();},
  };
}

test('30 idle minutes make only the initial health request',async()=>{
  const b=browser();await b.mount();await b.advance(30*60*1000);assert.equal(b.calls,1);
});
test('rapid tab visibility changes and simulation remounts reuse a recent check',async()=>{
  const b=browser();const unmount=await b.mount();
  for(let i=0;i<10;i++){await b.visible(false);await b.visible(true);}
  unmount();await b.mount();assert.equal(b.calls,1);
});
test('hidden pages do not check; returning after cache expiry checks once',async()=>{
  const b=browser({hidden:true});await b.mount();await b.advance(60*60*1000);assert.equal(b.calls,0);
  await b.visible(true);assert.equal(b.calls,1);await b.visible(false);await b.advance(6*60*1000);await b.visible(true);assert.equal(b.calls,2);
});
test('overlapping mounts share one request, including StrictMode remounts',async()=>{
  let resolve;const b=browser({request:()=>new Promise(r=>{resolve=r;})});
  const unmount=await b.mount();unmount();await b.mount();assert.equal(b.calls,1);
  resolve({ok:true});await b.advance(0);assert.equal(b.state,true);
});
test('manual retry recovers from offline and is deduplicated',async()=>{
  let ok=false;const b=browser({request:async()=>({ok})});await b.mount();assert.equal(b.state,false);
  ok=true;await b.retry();assert.equal(b.state,true);assert.equal(b.calls,2);
});
test('failed checks are cached during visibility changes and requests time out',async()=>{
  const b=browser({request:(_,options)=>new Promise((_,reject)=>options.signal.addEventListener('abort',()=>reject(new Error('aborted'))))});
  await b.mount();await b.advance(10000);assert.equal(b.state,false);
  await b.visible(false);await b.visible(true);assert.equal(b.calls,1);
});
