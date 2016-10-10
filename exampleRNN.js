// parameters for text generators
var sequenceLength = 10;
var sampleSize = 50;
var filePath = "input/simple.txt";
var samplePrime = "0";

// generic parameters
var running = false;
var iterationsPerFrame = 100;
var maxIterations = 2000;
var averageBatchTime = 0;
var totalIterations = 0;
var sil = null;
var layerSizes = [];

Stecy.setup = function () {
	
	Art.title = "Silicitect RRN demo";
	
};

Art.ready = function () {
	
	Stecy.loadFile(filePath, init);
	
	Art.doStyle(0, "whiteSpace", "pre-wrap", "font", "20px monospace", "tabSize", "4", "background", "#333", "color", "#ccc");
	
};

function init (e) {
	
	sil = new Silicitect();
	
	sil.parseText(e.responseText);
	
	layerSizes = [sil.text.characterSet.length, 10, 10, sil.text.characterSet.length];
	
	sil.reguliser = 1e-8;
	sil.learningRate = 0.1;
	sil.clipValue = 5;
	sil.decay = 0.95;
	sil.decayLinear = 0.9;
	sil.optimiser = Silicitect.adamOptimiser;
	
	sil.init(initLSTM);
	
	running = true;
	
}

function doNetworkStuff () {
	
	if (!running || totalIterations >= maxIterations) return;
	else totalIterations += iterationsPerFrame;
	
	sil.startLearningSession();
	
	for (var b = 0; b < iterationsPerFrame; b++) {
		
		trainCharacterSequence();
		
	}
	
	sil.endLearningSession();
	
	averageBatchTime += sil.batchTime;
	
	// Art.doClear(0);
	Art.doWrite(0, totalIterations + " " + (sil.totalLoss / iterationsPerFrame).toFixed(2) + " " + sil.batchTime + "ms");
	Art.doWrite(0, " avg " + (averageBatchTime / (totalIterations / iterationsPerFrame)).toFixed(0));
	
	printCharacterSequence();
	
}

Stecy.sequence("update", [doNetworkStuff]);

function trainCharacterSequence () {
	
	var sentence = sil.text.raw.substr(Math.floor(sil.randomUniform() * (sil.text.raw.length - sequenceLength)), sequenceLength);
	
	resetLastValues();
	
	for (var a = 0; a < sentence.length - 1; a++) {
		
		sil.matrixFillExcept(sil.net["inputLetters"], 0, sil.text.charToIndex[sentence.charAt(a)], 1);
		sil.matrixFillExcept(sil.net["desiredValues"], 0, sil.text.charToIndex[sentence.charAt(a + 1)], 1);
		
		updateLSTM();
		
		sil.computeLoss("output", "desiredValues", "matrixSoftmax", Silicitect.logLoss);
		
	}
	
	sil.backpropagate();
	
}

function printCharacterSequence () {
	
	Art.doWrite(0, " " + generateSentence(sampleSize, samplePrime) + "\n");
	
}

function generateSentence (length, prime) {
	
	var sentence = prime;
	
	resetLastValues();
	
	for (var a = 0; a < prime.length; a++) {
		
		var letter = sil.text.charToIndex[prime.charAt(a)];
		
		if (!(1 + letter)) continue;
		
		sil.matrixFillExcept(sil.net["inputLetters"], 0, letter, 1);
		
		updateLSTM();
		
	}
	
	resetLastValues();
	
	for (var a = 0; a < length; a++) {
		
		var letter = sil.text.charToIndex[sentence.charAt(sentence.length - 1)];
		
		sil.matrixFillExcept(sil.net["inputLetters"], 0, letter, 1);
		
		updateLSTM();
		
		var probabilities = sil.matrixSoftmax(sil.net["output"], 1.0);
		var index = sil.matrixSampleRandomSum(probabilities);
		
		sentence += sil.text.characterSet.charAt(index);
		
	}
	
	return sentence.slice(prime.length);
	
}

function resetLastValues () {
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		sil.matrixFill(sil.net["outputLast" + a], 0);
		
		if (sil.net["cellLast" + a]) {
			
			sil.matrixFill(sil.net["cellLast" + a], 0);
			
		}
		
	}
	
}

function initLSTM () {
	
	var net = sil.net;
	
	net["inputLetters"] = sil.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		net["input" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, prevSize), 0, 2);
		net["inputHidden" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, size), 0, 2);
		net["inputBias" + a] = sil.matrixFill(sil.matrix(size, 1), 1);
		
		net["forget" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, prevSize), 0, 2);
		net["forgetHidden" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, size), 0, 2);
		net["forgetBias" + a] = sil.matrixFill(sil.matrix(size, 1), 1);
		
		net["output" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, prevSize), 0, 2);
		net["outputHidden" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, size), 0, 2);
		net["outputBias" + a] = sil.matrixFill(sil.matrix(size, 1), 1);
		net["outputLast" + a] = sil.matrix(size, 1);
		
		net["cell" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, prevSize), 0, 2);
		net["cellHidden" + a] = sil.matrixRandomiseGaussian(sil.matrix(size, size), 0, 2);
		net["cellBias" + a] = sil.matrixFill(sil.matrix(size, 1), 1);
		net["cellLast" + a] = sil.matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = sil.matrixRandomiseGaussian(sil.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]), 0, 2);
	net["decoderBias"] = sil.matrixFill(sil.matrix(lastLayerSize, 1), 1);
	
	net["output"] = sil.matrix(lastLayerSize, 1);
	net["desiredValues"] = sil.matrix(lastLayerSize, 1);
	
}

function updateLSTM () {
	
	var net = sil.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["inputLetters"] : net["outputLast" + (a - 1)];
		
		var h0 = sil.matrixMultiply(net["input" + a], previousLayer);
		var h1 = sil.matrixMultiply(net["inputHidden" + a], net["outputLast" + a]);
		var inputGate = sil.matrixSigmoid(sil.matrixAdd(sil.matrixAdd(h0, h1), net["inputBias" + a]));
		
		var h2 = sil.matrixMultiply(net["forget" + a], previousLayer);
		var h3 = sil.matrixMultiply(net["forgetHidden" + a], net["outputLast" + a]);
		var forgetGate = sil.matrixSigmoid(sil.matrixAdd(sil.matrixAdd(h2, h3), net["forgetBias" + a]));
		
		var h4 = sil.matrixMultiply(net["output" + a], previousLayer);
		var h5 = sil.matrixMultiply(net["outputHidden" + a], net["outputLast" + a]);
		var outputGate = sil.matrixSigmoid(sil.matrixAdd(sil.matrixAdd(h4, h5), net["outputBias" + a]));
		
		var h6 = sil.matrixMultiply(net["cell" + a], previousLayer);
		var h7 = sil.matrixMultiply(net["cellHidden" + a], net["outputLast" + a]);
		var cellWrite = sil.matrixHyperbolicTangent(sil.matrixAdd(sil.matrixAdd(h6, h7), net["cellBias" + a]));
		
		var retain = sil.matrixElementMultiply(forgetGate, net["cellLast" + a]);
		var write = sil.matrixElementMultiply(inputGate, cellWrite);
		
		net["cellLast" + a] = sil.matrixAdd(retain, write);
		net["outputLast" + a] = sil.matrixElementMultiply(outputGate, sil.matrixHyperbolicTangent(net["cellLast" + a]));
		
	}
	
	net["output"] = sil.matrixAdd(sil.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]);
	
}

function initRNN () {
	
	var net = sil.net;
	
	net["inputLetters"] = sil.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		net["cell" + a] = sil.matrixRandomiseNormalised(sil.matrix(size, prevSize));
		net["cellHidden" + a] = sil.matrixRandomiseNormalised(sil.matrix(size, size));
		net["cellBias" + a] = sil.matrixFill(sil.matrix(size, 1), 1);
		
		net["outputLast" + a] = sil.matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = sil.matrixRandomiseNormalised(sil.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = sil.matrixFill(sil.matrix(lastLayerSize, 1), 1);
	
	net["output"] = sil.matrix(lastLayerSize, 1);
	net["desiredValues"] = sil.matrix(lastLayerSize, 1);
	
}

function updateRNN () {
	
	var net = sil.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["inputLetters"] : net["outputLast" + (a - 1)];
		
		var h0 = sil.matrixMultiply(net["cell" + a], previousLayer);
		var h1 = sil.matrixMultiply(net["cellHidden" + a], net["outputLast" + a]);
		var hiddenValues = sil.matrixRectifiedLinear(sil.matrixAdd(sil.matrixAdd(h0, h1), net["cellBias" + a]));
		
		net["outputLast" + a] = hiddenValues;
		
	}
	
	net["output"] = sil.matrixAdd(sil.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]);
	
	
}

function initGRU () {
	
	var net = sil.net;
	
	net["inputLetters"] = sil.matrixRandomiseUniform(sil.matrix(layerSizes[0], 1));
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		
		net["update" + a] = sil.matrixRandomiseNormalised(sil.matrix(layerSizes[a], prevSize));
		net["updateHidden" + a] = sil.matrixRandomiseNormalised(sil.matrix(layerSizes[a], layerSizes[a]));
		net["updateBias" + a] = sil.matrix(layerSizes[a], 1);
		
		net["reset" + a] = sil.matrixRandomiseNormalised(sil.matrix(layerSizes[a], prevSize));
		net["resetHidden" + a] = sil.matrixRandomiseNormalised(sil.matrix(layerSizes[a], layerSizes[a]));
		net["resetBias" + a] = sil.matrix(layerSizes[a], 1);
		
		net["cell" + a] = sil.matrixRandomiseNormalised(sil.matrix(layerSizes[a], prevSize));
		net["cellHidden" + a] = sil.matrixRandomiseNormalised(sil.matrix(layerSizes[a], layerSizes[a]));
		net["cellBias" + a] = sil.matrix(layerSizes[a], 1);
		
		net["outputLast" + a] = sil.matrix(layerSizes[a], 1);
		
	}
	
	net["decoder"] = sil.matrixRandomiseNormalised(sil.matrix(layerSizes[layerSizes.length - 1], layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = sil.matrix(layerSizes[layerSizes.length - 1], 1);
	
	net["output"] = sil.matrix(layerSizes[layerSizes.length - 1], 1);
	net["desiredValues"] = sil.matrix(layerSizes[layerSizes.length - 1], 1);
	
}

function updateGRU () {
	
	var net = sil.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["inputLetters"] : net["outputLast" + (a - 1)];
		
		var h0 = sil.matrixMultiply(net["update" + a], previousLayer);
		var h1 = sil.matrixMultiply(net["updateHidden" + a], net["outputLast" + a]);
		var updateGate = sil.matrixSigmoid(sil.matrixAdd(sil.matrixAdd(h0, h1), net["updateBias" + a]));
		
		var h2 = sil.matrixMultiply(net["reset" + a], previousLayer);
		var h3 = sil.matrixMultiply(net["resetHidden" + a], net["outputLast" + a]);
		var resetGate = sil.matrixSigmoid(sil.matrixAdd(sil.matrixAdd(h2, h3), net["resetBias" + a]));
		
		var h6 = sil.matrixMultiply(net["cell" + a], sil.matrixElementMultiply(previousLayer, resetGate));
		var h7 = sil.matrixMultiply(net["cellHidden" + a], net["outputLast" + a]);
		var cellWrite = sil.matrixHyperbolicTangent(sil.matrixAdd(sil.matrixAdd(h6, h7), net["cellBias" + a]));
		
		net["outputLast" + a] = sil.matrixAdd(sil.matrixElementMultiply(sil.matrixInvert(updateGate), cellWrite), sil.matrixElementMultiply(updateGate, net["outputLast" + a]));
		
	}
	
	net["output"] = sil.matrixAdd(this.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]);
	
}