FROM node:16-alpine

ARG NODE_ENV=development

ENV NODE_PATH=/srv/app/node_modules
WORKDIR /srv/app
COPY package.json ./

# RUN npm install

COPY . /srv/app

EXPOSE 3000
EXPOSE 35729

CMD ["npm", "start"]
