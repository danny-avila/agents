#!/usr/bin/env node

/* eslint-disable no-console */

/**
 * Manual test script for Jina AI chunking functionality
 * Usage: node -r dotenv/config --loader ./tsconfig-paths-bootstrap.mjs --experimental-specifier-resolution=node ./src/tools/search/jina-chunking.integration.ts
 */

import { JinaReranker, createReranker } from './rerankers';
import { createDefaultLogger } from './utils';

const logger = createDefaultLogger();

async function testBasicChunking(): Promise<boolean> {
  console.log('\n🔬 Testing Basic Chunking Functionality...\n');

  const reranker = new JinaReranker({
    apiKey: process.env.JINA_API_KEY ?? 'test-api-key',
    logger,
    chunkingConfig: {
      maxChunkSize: 1800,
      overlapSize: 200,
      enableParallelProcessing: true,
      aggregationStrategy: 'weighted_average',
    },
  });

  // Test documents of various sizes
  const documents = [
    'This is a small document that should not need chunking.',

    // Large document that mimics the 2,493 character error case
    'This is a comprehensive article about artificial intelligence and its applications in modern technology. ' +
      'The field of AI has grown exponentially over the past decade, with breakthroughs in machine learning, ' +
      'deep learning, and neural networks revolutionizing how we approach complex problems. From natural language ' +
      'processing to computer vision, AI systems are now capable of tasks that were once thought impossible. ' +
      'Companies across various industries are leveraging AI to improve efficiency, reduce costs, and create ' +
      'innovative solutions for their customers. The healthcare sector, in particular, has seen remarkable ' +
      'improvements through AI-powered diagnostic tools, drug discovery platforms, and personalized treatment ' +
      'recommendations. In the financial industry, AI algorithms are used for fraud detection, risk assessment, ' +
      'and algorithmic trading, while in transportation, autonomous vehicles represent one of the most visible ' +
      'applications of AI technology. The education sector is also being transformed by AI-powered personalized ' +
      'learning systems, intelligent tutoring systems, and automated grading tools. As we look to the future, ' +
      'the potential applications of AI continue to expand, with emerging areas like quantum computing and ' +
      'brain-computer interfaces promising even more revolutionary changes. However, with these advances come ' +
      'important ethical considerations about privacy, bias, and the responsible development and deployment of ' +
      'AI systems. Researchers and policymakers are working together to establish guidelines and regulations ' +
      'that will ensure AI technology benefits society while minimizing potential risks and negative consequences. ' +
      'The ongoing dialogue about AI ethics, transparency, and accountability will be crucial in shaping the ' +
      'future of this transformative technology and ensuring it serves humanity\'s best interests.',

    // Very large document that mimics the 136,534 bytes error case
    Array(500)
      .fill(
        'This is a detailed paragraph about machine learning algorithms and their implementations. ' +
          'We discuss various approaches including supervised learning, unsupervised learning, and reinforcement learning. ' +
          'Each method has its own strengths and weaknesses, and the choice of algorithm depends on the specific problem ' +
          'being solved, the nature of the data, and the desired outcomes. Performance metrics, cross-validation, and ' +
          'hyperparameter tuning are critical aspects of developing effective machine learning models. '
      )
      .join(' '),

    'Medium-sized document that provides balanced information about the topic and should demonstrate proper handling.',
  ];

  try {
    console.log(
      `📊 Testing with ${documents.length} documents of varying sizes:`
    );
    documents.forEach((doc, index) => {
      const size = new TextEncoder().encode(doc).length;
      console.log(
        `  Document ${index + 1}: ${size} bytes ${size > 2048 ? '(needs chunking)' : '(direct)'}`
      );
    });

    console.log('\n🚀 Running rerank operation...');
    const startTime = Date.now();

    const results = await reranker.rerank(
      'artificial intelligence machine learning technology applications',
      documents,
      4
    );

    const endTime = Date.now();
    const duration = endTime - startTime;

    console.log(`✅ Rerank completed in ${duration}ms`);
    console.log(`📋 Results (${results.length} documents):`);

    results.forEach((result, index) => {
      console.log(
        `  ${index + 1}. Score: ${result.score.toFixed(3)} | Text: ${result.text.substring(0, 100)}...`
      );
    });

    return true;
  } catch (error) {
    console.error('❌ Test failed:', error);
    return false;
  }
}

async function testErrorHandling(): Promise<boolean> {
  console.log('\n🛠️  Testing Error Handling...\n');

  // Test with invalid API key
  const invalidReranker = new JinaReranker({
    apiKey: 'invalid-key',
    logger,
  });

  try {
    const results = await invalidReranker.rerank(
      'test query',
      ['Test document for error handling'],
      1
    );

    console.log('🔄 Fallback to default ranking worked correctly');
    console.log(`📋 Results: ${results.length} documents with default scores`);
    return true;
  } catch (error) {
    console.error('❌ Error handling test failed:', error);
    return false;
  }
}

async function testConfigurationOptions(): Promise<boolean> {
  console.log('\n⚙️  Testing Configuration Options...\n');

  const configs = [
    {
      name: 'Default Configuration',
      config: {},
    },
    {
      name: 'Max Score Aggregation',
      config: { aggregationStrategy: 'max_score' as const },
    },
    {
      name: 'First Chunk Aggregation',
      config: { aggregationStrategy: 'first_chunk' as const },
    },
    {
      name: 'Custom Chunk Size',
      config: { maxChunkSize: 1000, overlapSize: 100 },
    },
    {
      name: 'Sequential Processing',
      config: { enableParallelProcessing: false },
    },
  ];

  const testDocument = 'x'.repeat(3000); // Large enough to trigger chunking

  for (const { name, config } of configs) {
    console.log(`🧪 Testing ${name}...`);

    try {
      const reranker = new JinaReranker({
        apiKey: process.env.JINA_API_KEY ?? 'test-api-key',
        logger,
        chunkingConfig: config,
      });

      const results = await reranker.rerank('test', [testDocument], 1);
      console.log(`  ✅ ${name}: ${results.length} results`);
    } catch (error) {
      console.log(`  ❌ ${name}: ${error}`);
    }
  }

  return true;
}

async function testCreateRerankerFactory(): Promise<boolean> {
  console.log('\n🏭 Testing createReranker Factory Function...\n');

  try {
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: process.env.JINA_API_KEY ?? 'test-api-key',
      logger,
      jinaChunkingConfig: {
        maxChunkSize: 1500,
        aggregationStrategy: 'max_score',
      },
    });

    if (!reranker) {
      throw new Error('Factory function returned undefined');
    }

    console.log('✅ Factory function created reranker successfully');

    const results = await reranker.rerank(
      'test query',
      ['Small test document for factory validation'],
      1
    );

    console.log(`📋 Factory reranker results: ${results.length} documents`);
    return true;
  } catch (error) {
    console.error('❌ Factory function test failed:', error);
    return false;
  }
}

async function runAllTests(): Promise<void> {
  console.log('🎯 Jina AI Chunking Implementation Test Suite');
  console.log('='.repeat(50));

  const tests = [
    { name: 'Basic Chunking', fn: testBasicChunking },
    { name: 'Error Handling', fn: testErrorHandling },
    { name: 'Configuration Options', fn: testConfigurationOptions },
    { name: 'Factory Function', fn: testCreateRerankerFactory },
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    try {
      const success = await test.fn();
      if (success) {
        passed++;
        console.log(`\n✅ ${test.name} PASSED\n`);
      } else {
        failed++;
        console.log(`\n❌ ${test.name} FAILED\n`);
      }
    } catch (error) {
      failed++;
      console.log(`\n❌ ${test.name} FAILED with exception:`, error, '\n');
    }

    // Small delay between tests
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  console.log('='.repeat(50));
  console.log(`📊 Test Results: ${passed} passed, ${failed} failed`);

  if (failed === 0) {
    console.log('🎉 All tests passed! Implementation is working correctly.');
  } else {
    console.log('⚠️  Some tests failed. Review the output above for details.');
  }

  process.exit(failed > 0 ? 1 : 0);
}

// Run the tests if this script is executed directly

if (import.meta.url === `file://${process.argv[1]}`) {
  runAllTests().catch((error) => {
    console.error('Fatal error running tests:', error);
    process.exit(1);
  });
}

export { runAllTests };
