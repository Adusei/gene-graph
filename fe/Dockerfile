FROM node:17-alpine
# set working direction
WORKDIR /app
# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install application dependencies
COPY package.json ./
COPY package-lock.json ./

RUN npm i

# add app
COPY . ./app

# start app
CMD ["npm", "start"]
