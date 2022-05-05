// https://towardsdatascience.com/build-a-simple-neural-network-with-tensorflow-js-d434a30fcb8

const tf = require('@tensorflow/tfjs-node');

const X = tf.tensor2d([
    // pink, small
    [0.1, 0.1],
    [0.3, 0.3],
    [0.5, 0.6],
    [0.4, 0.8],
    [0.9, 0.1],
    [0.75, 0.4],
    [0.75, 0.9],
    [0.6, 0.9],
    [0.6, 0.75]
]);


// 0 - no buy, 1 - buy
const y = tf.oneHot(tf.tensor1d([0, 0, 1, 1, 0, 0, 1, 1, 1], 'int32'), 2);
// printing the tensor
y.print();

const model = tf.sequential();


model.add(
    tf.layers.dense({
        inputShape: [2],
        units: 3,
        activation: "relu"
    })
);


model.add(
    tf.layers.dense({
        units: 2,
        activation: "softmax"
    })
);

model.compile({
    optimizer: tf.train.adam(0.1),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
});

async function train() {
    await model.fit(X, y, {
        shuffle: true,
        epochs: 20,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log("Epoch " + epoch);
                console.log("Loss: " + logs.loss + " accuracy: " + logs.acc);
            }
        }
    });
    return model;
}

train().then(model => {
    const predProb = model.predict(tf.tensor2d([[0.1, 0.6]])).dataSync();
    console.log('0:' + predProb[0]);
    console.log('1:' + predProb[1]);
});