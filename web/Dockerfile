# stage 0, "build-stage", based on Node.js, to build and compile the frontend
# FROM tiangolo/node-frontend as build-stage
FROM node:alpine as build-stage

WORKDIR /app
COPY package*.json /app/

# RUN npm install -g npm@latest
# RUN npm install
RUN yarn install

COPY ./ /app/

RUN npm run build

# stage 1, based on Nginx, to have only the compiled app, ready for production with Nginx
FROM nginx

COPY --from=build-stage /app/build/ /usr/share/nginx/html

# copy the default nginx.conf provided by tiangolo/node-frontend
# COPY --from=build-stage /nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage /app/nginx/nginx.conf /etc/nginx/conf.d/default.conf