const tf = require('@tensorflow/tfjs-node');

const model = tf.sequential();

// dense is a fully connected layer
const hidden = tf.layers.dense({
    units: 5, // number of nodes
    inputShape: [1], // input shape
    activation: 'relu',
});
model.add(hidden);

const hidden_2 = tf.layers.dense({
    units: 15,
    // here the input shape is "inferred from the previous shape"
    activation: 'elu',
});
model.add(hidden_2);


const output = tf.layers.dense({
    units: 2,
    // here the input shape is "inferred from the previous shape"
    activation: 'linear',
});
model.add(output);

// i'm done configuring the model so compile it
model.compile({
    optimizer: tf.train.adam(0.001),
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
    [170, 1], // predict tmr
    [160, -1],
    [160, 0],
    [161, 1],
    [162, 1]
]);

console.log(model.summary());

async function train() {

    for (let i = 0; i < 10; i++) {
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
})



