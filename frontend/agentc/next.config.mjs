/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['@langchain/core', '@langchain/community'],
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push({
        '@langchain/core': '@langchain/core',
        '@langchain/community': '@langchain/community',
      });
    }
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      net: false,
      tls: false,
    };
    return config;
  },
};

export default nextConfig;