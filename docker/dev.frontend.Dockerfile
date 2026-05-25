FROM node:22-alpine
WORKDIR /app

# Copy lockfiles first so npm ci is cached until deps change
COPY src/frontend/package.json src/frontend/package-lock.json ./

RUN npm ci

EXPOSE 3000

# dev:docker = "vite --host 0.0.0.0" — binds to all interfaces inside container
CMD ["npm", "run", "dev:docker"]
