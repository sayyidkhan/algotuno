// https://curiousily.com/posts/build-a-simple-neural-network-with-tensorflow-js-in-javascript/
const tf = require('@tensorflow/tfjs-node');

const ALPHA = 0.001;
const HIDDEN_SIZE = 4;

const DATA = tf.tidy(() => {
    return tf.tensor([
        // infections, infected countries
        [2.0, 1.0],
        [5.0, 1.0],
        [7.0, 4.0],
        [12.0, 5.0],
    ]);
});


const nextDayInfections = tf.tidy(() => {
    return tf.expandDims(tf.tensor([5.0, 7.0, 12.0, 19.0]), 1);
});

const nn_model = tf.tidy(() => {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            inputShape: [DATA.shape[1]],
            units: HIDDEN_SIZE,
            activation: "tanh",
        })
    );
    model.add(
        tf.layers.dense({
            units: HIDDEN_SIZE,
            activation: "tanh",
        })
    );
    model.add(
        tf.layers.dense({
            units: 1,
        })
    );

    model.compile({optimizer: tf.train.sgd(ALPHA), loss: "meanSquaredError"});
    return model;
});


//model.summary();
const train = async () => {
    const model = nn_model;

    await model.fit(DATA, nextDayInfections, {
        epochs: 200,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                if ((epoch + 1) % 10 === 0) {
                    console.log(`Epoch ${epoch + 1}: error: ${logs.loss}`)
                }
            },
        },
    });
    console.log(model.layers[0].getWeights()[0].shape);
    model.layers[0].getWeights()[0].print();
    console.log(model.layers[1].getWeights()[0].shape);
    model.layers[1].getWeights()[0].print();

    const lastDayFeatures = tf.tensor([[12.0, 5.0]]);
    model.predict(lastDayFeatures).print();
};


train().then(res => {
});