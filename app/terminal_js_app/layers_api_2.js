const tf = require('@tensorflow/tfjs-node');

const model = tf.sequential();

// dense is a fully connected layer
const hidden = tf.layers.dense({
    units: 5, // number of nodes
    inputShape: [1], // input shape
    activation: 'sigmoid',
});
model.add(hidden);

const hidden_2 = tf.layers.dense({
    units: 10,
    // here the input share is "inferred from the previous shape"
    activation: 'relu',
});
model.add(hidden_2);

const hidden_3 = tf.layers.dense({
    units: 10,
    // here the input share is "inferred from the previous shape"
    activation: 'relu6',
});
model.add(hidden_3);

const hidden_4 = tf.layers.dense({
    units: 10,
    // here the input share is "inferred from the previous shape"
    activation: 'sigmoid',
});
model.add(hidden_4);

const output = tf.layers.dense({
    units: 3,
    // here the input share is "inferred from the previous shape"
    activation: 'sigmoid',
});
model.add(output);

// i'm done configuring the model so compile it
model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: tf.losses.meanSquaredError,
});


// INPUT DATA
const xs = tf.tensor2d([
    [150],
    [170],
    [160],
    [160],
    [161],
]);

// OUTPUT DATA
const ys = tf.tensor2d([
    // down, up, same
    [0, 1, 0], // predict tmr
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
]);

async function train() {

    for (let i = 0; i < 10000; i++) {
        const config = {
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



