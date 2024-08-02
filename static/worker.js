import * as tf from "/dist/tf.min.js";
import * as ort from "/dist/ort.training.wasm.min.js";

// Set up wasm paths
ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;
// Worker code for message handling
self.addEventListener("message", async (event) => {
  let data = event.data;
  console.log(data);
  console.log(ort);
  console.log(tf);
  let userId = data.userId;
  // Get the subset of the dataset of the current user
  let dataset = data.dataset;

  // Get the base64 representation of images from json file, dict{ key, base64 }
  let base64data = await loadJson("/script/train_base64images.json");
  console.log(`CURRENTLY RUNNING USER ${userId}`);
  console.log(`LOADING TRAINING SESSION FOR USER ${userId}`);

  // Load the Training session of the current user
  await loadTrainingSession(ARTIFACTS_PATH);

  // Loop over the items of the dataset
  for (const [key, value] of dataset) {
    // Get the label predicted by the base model
    let label = await predict(base64data[key]);
    console.log(
      `Chat GPT predicted ${value}, ONNX model predicted ${
        Object.keys(label)[0]
      }`
    );
    // Compare the label predicted by the base model to the one predicted by chatgpt
    if (value !== Object.keys(label)[0]) {
      await train(base64data[key], value); // retrain the model on the output of chatgpt
    }
  }

  self.postMessage({
    userId: userId,
  });

  self.postMessage({
    epochMessage: "Model parameters updated",
    reload: true,
  });
});

self.onerror = function (error) {
  console.error("Worker error:", error);
};

let NUMEPOCHS = 2;
// Initialization of the paths

const ARTIFACTS_PATH = {
  checkpointState: "/artifacts/checkpoint",
  trainModel: "/artifacts/training_model.onnx",
  evalModel: "/artifacts/eval_model.onnx",
  optimizerModel: "/artifacts/optimizer_model.onnx",
};

let MODEL_PATH = "/model/base_model.onnx";

// Initialization of the inference session and the training session

let trainingSession = null;
let inferenceSession = null;

/**
 *  Instantiate an inference session
 * @async
 * @loadInferenceSession
 * @params {model_path}
 * @returns {Promise<void>}
 */
async function loadInferenceSession(MODEL_PATH) {
  console.log("Loading Inference Session");

  try {
    inferenceSession = await ort.InferenceSession.create(MODEL_PATH);
    console.log("Inference Session successfully loaded");
  } catch (err) {
    console.log("Error loading the Inference Session:", err);
    throw err;
  }
}

/**
 *  Instantiate a training session
 * @async
 * @loadTrainingSession
 * @params {model_path}
 * @returns {Promise<void>}
 */
async function loadTrainingSession(createOptions) {
  console.log("Trying to load Training Session");

  try {
    trainingSession = await ort.TrainingSession.create(createOptions);
    console.log("Training session loaded");
  } catch (err) {
    console.error("Error loading the training session:", err);
    throw err;
  }
}

/** 
 * Loads JSON from a given URL
 * @async
 * @load_Json
 * @params {model_path}
 * @returns {Promise<void>}
 */
async function loadJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const config = await response.json();
    return config;
  } catch (error) {
    console.error("Error loading config", error);
  }
}

const config = await loadJson("/script/config.json");
const pre = await loadJson("/script/preprocessor_config.json");

/**
 * Converts an image in base64 string format into a tensor of shape [1,3,224,224]
 * @async
 * @toTensorAndResize
 * @params {base64Data} the base64 encoded string representing the image
 * @returns {Promise<Tensor>}
 */
async function toTensorAndResize(base64Data) {
  const input_size = 224; // Example input size, adjust if necessary
  const shape = [1, 3, input_size, input_size];

  const imgBlob = await fetch(base64Data).then((res) => res.blob());
  const imgBitmap = await createImageBitmap(imgBlob);

  const canvas = new OffscreenCanvas(input_size, input_size);
  const ctx = canvas.getContext("2d");

  ctx.drawImage(
    imgBitmap,
    0,
    0,
    imgBitmap.width,
    imgBitmap.height,
    0,
    0,
    input_size,
    input_size
  );

  const resizedImageData = ctx.getImageData(0, 0, input_size, input_size);

  const dataFromImage = new Float32Array(3 * input_size * input_size);

  for (let i = 0; i < input_size * input_size; i++) {
    dataFromImage[i] = resizedImageData.data[i * 4]; // R
    dataFromImage[i + input_size * input_size] =
      resizedImageData.data[i * 4 + 1]; // G
    dataFromImage[i + 2 * input_size * input_size] =
      resizedImageData.data[i * 4 + 2]; // B
  }

  const imageTensor = new ort.Tensor("float32", dataFromImage, shape);
  return imageTensor;
}
/**
 * Normalize input image represented as a tensor
 * @param {tensor} tensor 
 * @returns {Tensor}
 */
async function preprocessImage(tensor) {
    const imageMean = pre.image_mean;
    const imageStd = pre.image_std;
  
    let data = await tensor.getData();
  
    data = data.map(function (value) {
      return (value / 255.0 - imageMean[0]) / imageStd[0];
    });
  
    let normalizedTensor = new ort.Tensor("float32", data, [1, 3, 224, 224]);
  
    return normalizedTensor;
  }

/** 
 * Performs softmax activation on logits in array format
 * @softmax
 * @params {logits} Raw outputs of the onnx model
 * @returns {Map} Probability distribution in an array
 */
function softmax(logits) {
  return logits.map(function (value, index) {
    return (
      Math.exp(value) /
      logits
        .map(function (y /*value*/) {
          return Math.exp(y);
        })
        .reduce(function (a, b) {
          return a + b;
        })
    );
  });
}
/** 
 * Sorts an array in descending order
 * @argsort
 * @params {array} The array to be sorted
 * @returns {Promise<Array>} The sorted array
 */
function argsort(array) {
  const arrayWithIndices = Array.from(array).map((value, index) => ({
    value,
    index,
  }));

  arrayWithIndices.sort((a, b) => b.value - a.value);

  return arrayWithIndices.map((item) => item.index);
}

/**
 * Creates the expected logits output for a given class
 * @createTargetTensor
 * @params {new_class} The correct class of the image
 * @returns {Tensor} A tensor with a highest value at the index of the correct class
 */
function createTargetTensor(new_class) {
  const index = config.label2id[new_class];
  const shape = [1, 1000];
  const low_value = -3.5;

  const size = shape.reduce((a, b) => a * b);
  let data = new Float32Array(size).fill(low_value);
  data[index] = -low_value;

  return new ort.Tensor("float32", data, shape);
}

/**
 * Performs data augmentation on a given image
 * @augmentImage
 * @params {image}
 * @returns {Tensor}
 */
export function augmentImage(image) {
  let augmentedImage = tf.expandDims(image, 0);

  if (Math.random() > 0.5) {
    augmentedImage = tf.image.flipLeftRight(augmentedImage);
  }

  const rotations = [0, 90, 180, 270];
  const angle = rotations[Math.floor(Math.random() * rotations.length)];
  augmentedImage = tf.image.rotateWithOffset(augmentedImage, angle / 90);

  return augmentedImage;
}

/** 
 * Preprocesses the given image for training
 * @async
 * @preprocessImageTraining
 * @params {base64Data} The base64 encoded string representing the image
 * @params {pre} The json file containing the preprocessing parameters
 * @returns {Promise<Set[Tensor]>}
 */
async function preprocessImageTraining(base64Data, pre) {
  const numImages = 7;
  const images = [];
  const inputSize = {
    width: pre.size.width,
    height: pre.size.height,
  };

  let image = await toTensorAndResize(base64Data);
  const imageData_ = await image.getData();

  for (let i = 0; i < numImages; i++) {
    let imageTensor = tf.tensor(imageData_, [3, 224, 224], "float32");

    imageTensor = tf.transpose(imageTensor, [1, 2, 0]);

    let augmentedTensor = augmentImage(imageTensor);

    augmentedTensor = tf.transpose(augmentedTensor, [0, 3, 1, 2]);

    let data_ = await augmentedTensor.data();
    let shape = augmentedTensor.shape;

    augmentedTensor = new ort.Tensor("float32", data_, shape);

    augmentedTensor = await preprocessImage(augmentedTensor);

    images.push(augmentedTensor);
  }

  return images;
}

/** 
 * Runs inference on a given image in base64 encoded string format
 * @async
 * @predict
 * @params {base64Data}
 * @returns {Promise<List[String]>}
 */
async function predict(base64Data) {
  if (!inferenceSession) {
    console.log("Inference session not loaded yet, waiting...");
    await loadInferenceSession(MODEL_PATH);
  }

  let imageTensor = await toTensorAndResize(base64Data);
  let inputImage = await preprocessImage(imageTensor);

  let feeds = {
    pixel_values: inputImage,
  };

  let results = await inferenceSession.run(feeds);
  let logits = results["logits"];

  let probs = softmax(logits.cpuData);
  let sorted_indices = argsort(probs);

  const id2label = config.id2label;

  let labels = {};
  sorted_indices.slice(0, 7).forEach((i, x) => {
    labels[id2label[i.toString()]] = probs[i];
  });

  return labels;
}

/** 
 * Stop the current thread for a given time
 * @sleep
 * @params {ms} Integer
 * @returns {Promise<void>}
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 *  Trains the model on a given image with a given label
 * @train
 * @params {base64} String - base64 encodeed representation of the image
 * @params {new_class} String - the true label of the image
 * @returns {Promise<void>}
 */
async function train(base64, new_classe) {
  let target_tensor = createTargetTensor(new_classe);
  let pre = await loadJson("/script/preprocessor_config.json");
  let images = await preprocessImageTraining(base64, pre);

  images = images.map((img) => new ort.Tensor(img.type, img.data, img.dims));

  const startTrainingTime = Date.now();
  console.log("Training started");

  for (let epoch = 0; epoch < NUMEPOCHS; epoch++) {
    await runTrainingEpoch(images, epoch, target_tensor);
  }

  const trainingTime = Date.now() - startTrainingTime;
  console.log(`Training completed in ${trainingTime}`);
}

async function runTrainingEpoch(images, epoch, target_tensor) {
  const epochStartTime = Date.now();
  const lossNodeName = trainingSession.handler.outputNames[0];

  console.log(
    `TRAINING | Epoch ${epoch + 1} / ${NUMEPOCHS} | Starting Training ... `
  );

  for (let image of images) {
    const feeds = {
      pixel_values: image,
      target: target_tensor,
    };

    const results = await trainingSession.runTrainStep(feeds);
    const loss = results[lossNodeName].data;

    console.log(`LOSS: ${loss}`);

    await trainingSession.runOptimizerStep();
    await trainingSession.lazyResetGrad();
  }

  const epochTime = Date.now() - epochStartTime;
  console.log(`Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`);
}