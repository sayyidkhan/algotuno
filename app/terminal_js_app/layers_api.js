const tf = require('@tensorflow/tfjs-node');

const model = tf.sequential();

// dense is a fully connected layer
const hidden = tf.layers.dense({
    units: 4, // number of nodes
    inputShape: [2], // input shape
    activation: 'sigmoid',
});
model.add(hidden);

const output = tf.layers.dense({
    units: 1,
    // here the input share is "inferred from the previous shape"
    activation: 'sigmoid',
});
model.add(output);

// i'm done configuring the model so compile it
model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: tf.losses.meanSquaredError,
});


// INPUT DATA
const xs = tf.tensor2d([
    [0, 0],
    [0.5, 0.5],
    [1, 1],
]);

// OUTPUT DATA
const ys = tf.tensor2d([
    [1],
    [0.5],
    [0],
]);

async function train() {

    for (let i = 0; i < 1000; i++) {
        const config = {
            shuffle : true,
            epochs: 10,
        };
        const response = await model.fit(xs, ys, config);
        console.log(response.history.loss[0]);
    }

}


// do the train first before doing the prediction
train().then(() => {
    console.log("training complete");
    let outputs = model.predict(xs);
    outputs.print();
});



