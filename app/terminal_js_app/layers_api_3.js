const tf = require('@tensorflow/tfjs-node');
const {plot, Plot} = require('nodeplotlib');

const model = tf.sequential();

// dense is a fully connected layer
const hidden = tf.layers.inputLayer({
    units: 52, // number of nodes
    inputShape: [1], // input shape
});
model.add(hidden);

const hidden_2 = tf.layers.dense({
    units: (52 * 3),
    // here the input shape is "inferred from the previous shape"
    activation: 'elu',
});
model.add(hidden_2);

const hidden_3 = tf.layers.dense({
    units: (52 * 3),
    // here the input shape is "inferred from the previous shape"
    activation: 'tanh',
});
model.add(hidden_3);


const output = tf.layers.dense({
    units: 2,
    // here the input shape is "inferred from the previous shape"
    activation: 'linear',
});
model.add(output);

// i'm done configuring the model so compile it
model.compile({
    optimizer: tf.train.adam(0.00005),
    loss: tf.losses.meanSquaredError,
});


// INPUT DATA
const xs = tf.tensor2d([
    [96.96],
    [97.13],
    [101.42],
    [97.34],
    [94.02],
    [93.99],
    [96.04],
    [96.91],
    [103.01],
    [102.26],
    [105.92],
    [105.67],
    [109.99],
    [108.66],
    [109.85],
    [105.68],
    [93.74],
    [92.72],
    [90.52],
    [95.22],
    [100.35],
    [97.92],
    [98.83],
    [95.33],
    [93.4],
    [95.89],
    [96.68],
    [98.78],
    [98.66],
    [104.21],
    [107.48],
    [108.18],
    [109.36],
    [106.94],
    [107.73],
    [103.13],
    [114.92],
    [112.71],
    [113.05],
    [114.06],
    [117.63],
    [116.6],
    [113.72],
    [108.84],
    [108.43],
    [110.06],
    [111.79],
    [109.9],
    [113.95],
    [115.97],
    [116.52],
    [115.82]
]);

// OUTPUT DATA
const ys = tf.tensor2d(
    // 1 (up), 0 (hold), -1 (down)
    [
        [97.13, 1], [101.42, 1], [97.34, -1],
        [94.02, -1], [93.99, -1], [96.04, 1],
        [96.91, 0], [103.01, 1], [102.26, -1],
        [105.92, 1], [105.67, 0], [109.99, 1],
        [108.66, -1], [109.85, 1], [105.68, -1],
        [93.74, -1], [92.72, -1], [90.52, -1],
        [95.22, 1], [100.35, 1], [97.92, -1],
        [98.83, 1], [95.33, -1], [93.4, -1],
        [95.89, 1], [96.68, 1], [98.78, 1],
        [98.66, 0], [104.21, 1], [107.48, 1],
        [108.18, 1], [109.36, 1], [106.94, -1],
        [107.73, 1], [103.13, -1], [114.92, 1],
        [112.71, -1], [113.05, 1], [114.06, 1],
        [117.63, 1], [116.6, -1], [113.72, -1],
        [108.84, -1], [108.43, 0], [110.06, 1],
        [111.79, 1], [109.9, -1], [113.95, 1],
        [115.97, 1], [116.52, 1], [115.82, -1],
        [117.91, 1]
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
});





