# Single-stage image running Next dev on 8080
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ENV PORT=8080
EXPOSE 8080
CMD ["npm", "run", "dev", "--", "-p", "8080", "-H", "0.0.0.0"]
