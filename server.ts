import Fastify, { FastifyInstance } from 'fastify';
import { TypeBoxTypeProvider } from '@fastify/type-provider-typebox';
import dotenv from 'dotenv';
import { agentsRouter } from './routes/agents';

// Load environment variables
dotenv.config();

const server: FastifyInstance = Fastify({
  logger: true
}).withTypeProvider<TypeBoxTypeProvider>();

// Register routes
server.register(agentsRouter, { prefix: '/agents' });

const start = async () => {
  try {
    const host = process.env.HOST || 'localhost';
    const port = parseInt(process.env.PORT || '3000', 10);
    
    await server.listen({ host, port });
    console.log(`Server is running on ${host}:${port}`);
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
};

start();
