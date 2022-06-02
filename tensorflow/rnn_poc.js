const { tensor_util } = require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs-node');

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
let dataX =[], dataY = [];

// Creating the data
for(let i=0; i<alphabet.length-1; i++) {
    dataX.push([[parseInt(alphabet.indexOf(alphabet.charAt(i)))]]);
    // One-hot-encoding the output values
    let arr = new Array(alphabet.length).fill(0)
    arr[alphabet.indexOf(alphabet.charAt(i+1))] = 1;
    dataY.push(arr);
}

// Transforming the data to tensors
const x = tf.tensor(dataX);
const y = tf.tensor(dataY);

// Printing the tensors
x.print()
y.print()

// Creating the RNN Model
const model = tf.sequential();
model.add(tf.layers.lstm({units:32, inputShape:[1, 1]}))
model.add(tf.layers.dense({units:32, activation:'relu'}));
model.add(tf.layers.dense({units:alphabet.length, activation:'softmax'}));

// Compiling the model
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

// Fitting the model
model.fit(x, y, {
    batchSize: alphabet.length,
    epochs: 10
}).then((history) => {
    // printing loss and predictions
    console.log(history.history.loss);
    console.log(model.predict(x).argMax(1).dataSync())
})