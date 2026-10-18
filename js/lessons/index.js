import data from './data.js';
import intro from './intro.js';
import gradientDescent from './gradient-descent.js';
import overfitting from './overfitting.js';
import supervisedProcess from './supervised-process.js';
import linearRegression from './linear-regression.js';
import logisticRegression from './logistic-regression.js';
import decisionTree from './decision-tree.js';
import randomForest from './random-forest.js';
import svm from './svm.js';
import knn from './knn.js';
import naiveBayes from './naive-bayes.js';
import gradientBoosting from './gradient-boosting.js';
import kmeans from './kmeans.js';
import pca from './pca.js';
import neuralNetwork from './neural-network.js';
import cnn from './cnn.js';
import attention from './attention.js';
import qLearning from './q-learning.js';

// Order here is the order shown in the sidebar; lessons with the same
// category should stay adjacent so they group under one heading.
export const lessons = [
  data,
  intro,
  gradientDescent,
  overfitting,
  supervisedProcess,
  linearRegression,
  logisticRegression,
  decisionTree,
  randomForest,
  svm,
  knn,
  naiveBayes,
  gradientBoosting,
  kmeans,
  pca,
  neuralNetwork,
  cnn,
  attention,
  qLearning,
];
