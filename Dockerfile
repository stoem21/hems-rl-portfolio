FROM node:22-alpine AS frontend-build
WORKDIR /build/hems-frontend
COPY hems-frontend/package*.json ./
RUN npm ci
COPY hems-frontend/ ./
ARG VITE_BACKEND_URL=""
ARG VITE_SITE_BASE="/"
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL} VITE_SITE_BASE=${VITE_SITE_BASE}
RUN npm run build

# The default build is a code-only dashboard. No thesis/data/checkpoints are copied.
FROM nginx:alpine AS dashboard
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /build/hems-backend/dist /usr/share/nginx/html
EXPOSE 80
