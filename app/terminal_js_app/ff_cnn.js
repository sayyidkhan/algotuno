// dataset
// https://github.com/gpbl/SwiftChart/blob/master/Example/SwiftChart/AAPL.json

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

let rawdata = fs.readFileSync('appl_price.json');
let data = JSON.parse(rawdata);


// turn the date into 3 dimensions
// one dimension the year
// one dimension the month
// one dimension the day

/**
 *  Data sets
 *
 *  {"Date":"1980-12-08","Open":0.513393,"High":0.515625,"Low":0.513393,"Close":0.513393,"Adj Close":0.023268,"Volume":117258400
 */

const d = {
    'test_times': [new Date('1995-12-17T00:00:00.000').getTime(),
        new Date('2018-06-25T00:00:00.000').getTime()],
    'test_highs': [184.160004, 184.919998]
};

const prep = function (data) {
    return new Promise(function (resolve, reject) {
        let dates = [], highs = [];
        try {
            for (let i = 0; i < data.length; i++) {
                dates.push(new Date(data[i]['Date'] + 'T00:00:00.000').getTime())
                highs.push(data[i]['Close'])
            }
        } catch (ex) {
            resolve(ex)
            console.log(ex)
        }
        return resolve({
            dates: dates,
            highs: highs
        })
    })
};

const modelHelper = function (model) {
    let layerNames = ['conv1d_Conv1D1',
        'max_pooling1d_MaxPooling1D1',
        'conv1d_Conv1D2',
        'max_pooling1d_MaxPooling1D2',
        'dense_Dense1'
    ];

    console.log("MODEL SUMMARY: ");
    model.summary();

    console.log("MODEL LAYERS: ");
    for (let i = 0; i < layerNames.length; i++) {
        console.log(model.getLayer(layerNames[i]));
    }

    console.log("SHAPES BY LAYER: ");
    for (let i = 0; i < layerNames.length; i++) {
        console.log("INPUT SHAPE " + model.getLayer(layerNames[i]).input.shape + " OUTPUT SHAPE " + model.getLayer(layerNames[i]).output.shape)
    }

};

// Step 1 - build Convolutional network
const buildCnn = function (data) {
    //A promise represents the eventual result of an asynchronous
    //operation. It is a placeholder into which the successful
    //result value or reason for failure will materialize.
    return new Promise(function (resolve, reject) {
        //Linear stack of layers.
        const model = tf.sequential();

        //This layer creates a convolution kernel
        // that is convolved (actually cross-correlated)
        // with the layer input to produce a tensor of outputs.

        //kernel size - An integer or tuple/list of a single integer,
        // specifying the length of the 1D convolution window.
        //filters - Integer, the dimensionality of the output space
        // (i.e. the number of filters in the convolution).
        //stride- An integer or tuple/list of a single integer,
        // specifying the stride length of the convolution.
        //activation- nonlinearity
        // kernel-init - An initializer for the bias vector.
        //variance scaling - the weights initialization technique that tries to make the variance of the outputs
        //of a layer to be equal to the variance of its inputs
        model.add(tf.layers.conv1d({
            inputShape: [data.dates.length, 1],
            kernelSize: 100,
            filters: 8,
            strides: 2,
            activation: 'relu',
            kernelInitializer: 'VarianceScaling'
        }));

//poolsize - An integer or tuple/list of a single integer,
//representing the size of the pooling window.
//strides: An integer or tuple/list of a single integer,
//specifying the strides of the pooling operation.
        model.add(tf.layers.maxPooling1d({
            poolSize: [500],
            strides: [2]
        }));

        model.add(tf.layers.conv1d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'VarianceScaling'
        }));

        model.add(tf.layers.maxPooling1d({
            poolSize: [100],
            strides: [2]
        }));

        //dense (also known as a fully connected layer),
        //which will perform the final classification.
        // Flattening the output of a convolution+pooling layer pair before a dense layer is another common pattern in neural networks:
        model.add(tf.layers.dense({
            units: 10,
            kernelInitializer: 'VarianceScaling',
            activation: 'softmax'
        }));

        //The Promise.resolve(value) method returns a
        //Promise object that is resolved with the given value.
        return resolve({
            'model': model,
            'data': data
        });
    })
};

//Step 2 Train Model
const cnn = function (model, data, cycles) {
    const tdates = tf.tensor1d(data.dates),
        thighs = tf.tensor1d(data.highs),
        test = tf.tensor1d(d.test_times),
        out = model.getLayer('dense_Dense1');

    //console.log(tdates)
    //console.log(thighs)

    //modelHelper(model);

    //console.log(tdates.reshape([1, 1960, 1]))
    //console.log(thighs.reshape([1, 1960, 1]))

    return new Promise(function (resolve, reject) {
        setTimeout(function () {
            try {
                model.compile({optimizer: 'sgd', loss: 'binaryCrossentropy', lr: 0.1});
                model.fit(
                    tdates.reshape([1, 1960, 1]),
                    thighs.reshape([1, 1960, 1]), {
                        batchSize: 3,
                        epochs: cycles
                    }).then(function () {
                    console.log('');
                    console.log('Running CNN for AAPL at ' + cycles + ' epochs');
                    console.log(model.predict(test));
                    console.log(d.test_highs);
                    resolve(console.log(''));
                });
            } catch (ex) {
                resolve(console.log(ex));
            }
        }, 5000)
    });
};


prep(data).then(function (result) {
    buildCnn(result).then(function (built) {
        cnn(built.model, built.data, 100).then(function (e) {
            console.log('Completed tests at ' + new Date() + '... thanks for waiting!')
        })
    });
});
