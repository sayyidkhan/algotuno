const tf = require('@tensorflow/tfjs-node');

// Creating dataset
const x = tf.randomUniform([100], -1, 1);
const y = x.mul(0.5).add(tf.randomUniform([100], -0.1, 0.1));
x.print();
y.print();

// Creating variables
const m = tf.variable(tf.scalar(Math.random()*2-1));
const b = tf.variable(tf.scalar(Math.random()*2-1));

// Specifying a learning rate and an optimizer
const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);

// Mean Squared Error Loss
function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function predict(xs) {
  // y = mx+b
  const ys = xs.mul(m).add(b)
  return ys
}

// Training model for 50 epochs
for(let i = 0; i < 50; i++) {
  optimizer.minimize(() => loss(predict(x), y))
}

// Printing our weights (slope, intercept)
m.print()
b.print()