/*Silicitect, model neural network architectures in JavaScript for silicon hardware.
Copyright (C) 2016 Fjodor van Veen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>*/

// parameters for text generators
var letterCount = 10;
var sampleSize = 50;
var filePath = "input/simple.txt";
var samplePrime = "0";

// generic parameters
var running = false;
var iterationsPerFrame = 100;
var maxIterations = 2e3;
var averageBatchTime = 0;
var totalIterations = 0;
var seed = 3;

// global variables for the examples
var sil = null;
var textParser = null;
var layerSizes = [];
var examples = {};

Stecy.setup = function () {
	
	Art.title = "Silicitect";
	
	Art.width = 500;
	Art.height = 100;
	Art.useCanvas = true;
	Art.stretch = 2;
	
};

Art.ready = function () {
	
	Stecy.loadFile(filePath, init);
	
	Art.doStyle(0, "whiteSpace", "pre-wrap", "font", "20px monospace", "tabSize", "6", "background", "#333", "color", "#ccc");
	
};

function init (e) {
	
	textParser = new TextParser(e.responseText, "");
	layerSizes = [textParser.chars.length, 10, 10, textParser.chars.length];
	// layerSizes = [2, 2, 1];
	// layerSizes = [5, 4, 3, 4, 5];
	
	sil = new Silicitect(examples.initLSTM, examples.updateLSTM);
	
	sil.reguliser = 1e-5;
	sil.learningRate = 0.1;
	sil.clipValue = 5;
	sil.decay = 0.9;
	sil.decayLinear = 0.9;
	sil.optimiser = Silicitect.adamOptimiser;
	
	Matrix.silicitect = sil;
	
	running = true;
	
}

function doNetworkStuff () {
	
	if (!running || totalIterations >= maxIterations) return;
	else totalIterations += iterationsPerFrame;
	
	sil.startLearningSession();
	
	for (var b = 0; b < iterationsPerFrame; b++) {
		
		// trainAutoencoder();
		// trainLogicGate();
		trainCharacterSequence();
		
	}
	
	sil.endLearningSession();
	
	averageBatchTime += sil.batchTime;
	
	// Art.doClear(0);
	Art.doWrite(0, totalIterations + " " + (sil.totalLoss / iterationsPerFrame).toFixed(2) + " " + sil.batchTime + "ms");
	Art.doWrite(0, " avg " + (averageBatchTime / (totalIterations / iterationsPerFrame)).toFixed(0));
	
	// drawAutoencoder();
	// printAutoencoder();
	// printLogicGate();
	printCharacterSequence();
	
}

Stecy.sequence("update", [doNetworkStuff]);

function trainLogicGate () {
	
	Matrix.w[sil.network["input"].i + 0] = +(Random.uniform() > 0.5);
	Matrix.w[sil.network["input"].i + 1] = +(Random.uniform() > 0.5);
	Matrix.w[sil.network["desiredValues"].i] = +(Matrix.w[sil.network["input"].i + 0] + Matrix.w[sil.network["input"].i + 1] > 0);
	sil.update();
	sil.computeLoss("output", "desiredValues", Matrix.nothing, Silicitect.linearLoss);
	sil.backpropagate();
	
}

function printLogicGate () {
	
	Matrix.w[sil.network["input"].i + 0] = 0;
	Matrix.w[sil.network["input"].i + 1] = 0;
	sil.update();
	Art.doWrite(0, "\n00 " + Matrix.w[sil.network["output"].i].toFixed(2) + "\n");
	Matrix.w[sil.network["input"].i + 0] = 0;
	Matrix.w[sil.network["input"].i + 1] = 1;
	sil.update();
	Art.doWrite(0, "01 " + Matrix.w[sil.network["output"].i].toFixed(2) + "\n");
	Matrix.w[sil.network["input"].i + 0] = 1;
	Matrix.w[sil.network["input"].i + 1] = 0;
	sil.update();
	Art.doWrite(0, "10 " + Matrix.w[sil.network["output"].i].toFixed(2) + "\n");
	Matrix.w[sil.network["input"].i + 0] = 1;
	Matrix.w[sil.network["input"].i + 1] = 1;
	sil.update();
	Art.doWrite(0, "11 " + Matrix.w[sil.network["output"].i].toFixed(2) + "\n");
	
}

function trainAutoencoder () {
	
	var set = Random.uniform() < 0.5 ? [0, 0, 0, 0, 1] : [1, 1, 1, 1, 0];
	
	for (var a = 0; a < sil.network["input"].l; a++) {
		
		Matrix.w[sil.network["input"].i + a] = set[a];
		
	}
	
	sil.update();
	sil.computeLoss("output", "input", Matrix.nothing, Silicitect.linearLoss);
	sil.backpropagate();
	
}

function printAutoencoder () {
	
	for (var a = 0; a < sil.network["input"].l; a++) {
		
		Matrix.w[sil.network["input"].i + a] = +(Random.uniform() < 0.5);
		
	}
	
	sil.update();
	
	Art.doWrite(0, " ");
	
	for (var a = 0; a < sil.network["input"].l; a++) {
		
		Art.doWrite(0, Matrix.w[sil.network["input"].i + a].toFixed(0));
		
	}
	
	Art.doWrite(0, " ");
	
	for (var a = 0; a < sil.network["output"].l; a++) {
		
		Art.doWrite(0, Matrix.w[sil.network["output"].i + a].toFixed(0));
		
	}
	
	Art.doWrite(0, "\n");
	
}

function drawAutoencoder () {
	
	Art.canvas.clearRect(0, 0, Art.width, Art.height);
	
	for (var a = 0; a < sil.network["input"].l; a++) {
		
		sil.network["input"].w[a] = +(Random.uniform() < 0.5);
		
	}
	
	for (var a = 0; a < sil.network["input"].l; a++) {
		
		var value = sil.network["input"].w[a];
		
		Art.canvas.fillStyle = "rgb(255, " + Math.floor(value * 256) + ", 0)";
		Art.canvas.fillRect(a * 100, 0, 100, 50);
		
	}
	
	for (var a = 0; a < sil.network["output"].l; a++) {
		
		var value = sil.network["output"].w[a];
		
		Art.canvas.fillStyle = "rgb(255, " + Math.floor(value * 256) + ", 0)";
		Art.canvas.fillRect(a * 100, 50, 100, 50);
		
	}
	
	sil.update();
	
}

function trainCharacterSequence () {
	
	var sentence = textParser.text.substr(Math.floor(Random.uniform() * (textParser.text.length - letterCount)), letterCount);
	
	resetLastValues();
	
	for (var a = 0; a < sentence.length - 1; a++) {
		
		sil.network["inputLetters"].fillExcept(0, textParser.charToIndex[sentence.charAt(a)], 1);
		sil.network["desiredValues"].fillExcept(0, textParser.charToIndex[sentence.charAt(a + 1)], 1);
		sil.update();
		sil.computeLoss("output", "desiredValues", Matrix.softmax, Silicitect.logLoss);
		
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
		
		var letter = textParser.charToIndex[prime.charAt(a)];
		
		if (!(1 + letter)) continue;
		
		sil.network["inputLetters"].fillExcept(0, letter, 1);
		sil.update();
		
	}
	
	resetLastValues();
	
	for (var a = 0; a < length; a++) {
		
		var letter = textParser.charToIndex[sentence.charAt(sentence.length - 1)];
		
		sil.network["inputLetters"].fillExcept(0, letter, 1);
		
		sil.update();
		
		var probabilities = Matrix.softmax(sil.network["output"], 1.0);
		var index = Matrix.sampleRandomSum(probabilities);
		
		sentence += textParser.chars.charAt(index);
		
	}
	
	return sentence.slice(prime.length);
	
}

function resetLastValues () {
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		sil.network["outputLast" + a].fill(0);
		
		if (sil.network["cellLast" + a]) {
			
			sil.network["cellLast" + a].fill(0);
			
		}
		
	}
	
}

examples.initFF = function () {
	
	var network = this.network;
	
	network["input"] = new Matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		network["hidden" + a] = new Matrix(size, prevSize).randomiseNormalised();
		network["hiddenBias" + a] = new Matrix(size, 1).fill(1);
		
		network["outputLast" + a] = new Matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	network["decoder"] = new Matrix(lastLayerSize, layerSizes[layerSizes.length - 2]).randomiseNormalised();
	network["decoderBias"] = new Matrix(lastLayerSize, 1).fill(1);
	
	network["output"] = new Matrix(lastLayerSize, 1);
	network["desiredValues"] = new Matrix(lastLayerSize, 1);
	
}

examples.updateFF = function () {
	
	var network = this.network;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? network["input"] : network["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(network["hidden" + a], previousLayer);
		
		network["outputLast" + a] = Matrix.sigmoid(Matrix.add(h0, network["hiddenBias" + a]));
		
	}
	
	network["output"] = Matrix.sigmoid(Matrix.add(Matrix.multiply(network["decoder"], network["outputLast" + (layerSizes.length - 2)]), network["decoderBias"]));
	
}

examples.initAE = function () {
	
	var network = this.network;
	
	network["input"] = new Matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		network["hidden" + a] = new Matrix(size, prevSize).randomiseNormalised();
		network["hiddenBias" + a] = new Matrix(size, 1).fill(1);
		
		network["outputLast" + a] = new Matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	network["decoder"] = new Matrix(lastLayerSize, layerSizes[layerSizes.length - 2]).randomiseNormalised();
	network["decoderBias"] = new Matrix(lastLayerSize, 1).fill(1);
	
	network["output"] = new Matrix(lastLayerSize, 1);
	
}

examples.updateAE = function () {
	
	var network = this.network;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? network["input"] : network["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(network["hidden" + a], previousLayer);
		
		network["outputLast" + a] = Matrix.sigmoid(Matrix.add(h0, network["hiddenBias" + a]));
		
	}
	
	network["output"] = Matrix.sigmoid(Matrix.add(Matrix.multiply(network["decoder"], network["outputLast" + (layerSizes.length - 2)]), network["decoderBias"]));
	
}

examples.initLSTM = function () {
	
	var network = this.network;
	
	network["inputLetters"] = new Matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		network["input" + a] = new Matrix(size, prevSize).randomiseNormalised();
		network["inputHidden" + a] = new Matrix(size, size).randomiseNormalised();
		network["inputBias" + a] = new Matrix(size, 1).fill(1);
		
		network["forget" + a] = new Matrix(size, prevSize).randomiseNormalised();
		network["forgetHidden" + a] = new Matrix(size, size).randomiseNormalised();
		network["forgetBias" + a] = new Matrix(size, 1).fill(1);
		
		network["output" + a] = new Matrix(size, prevSize).randomiseNormalised();
		network["outputHidden" + a] = new Matrix(size, size).randomiseNormalised();
		network["outputBias" + a] = new Matrix(size, 1).fill(1);
		network["outputLast" + a] = new Matrix(size, 1);
		
		network["cell" + a] = new Matrix(size, prevSize).randomiseNormalised();
		network["cellHidden" + a] = new Matrix(size, size).randomiseNormalised();
		network["cellBias" + a] = new Matrix(size, 1).fill(1);
		network["cellLast" + a] = new Matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	network["decoder"] = new Matrix(lastLayerSize, layerSizes[layerSizes.length - 2]).randomiseNormalised();
	network["decoderBias"] = new Matrix(lastLayerSize, 1).fill(1);
	
	network["output"] = new Matrix(lastLayerSize, 1);
	network["desiredValues"] = new Matrix(lastLayerSize, 1);
	
}

examples.updateLSTM = function () {
	
	var network = this.network;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? network["inputLetters"] : network["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(network["input" + a], previousLayer);
		var h1 = Matrix.multiply(network["inputHidden" + a], network["outputLast" + a]);
		var inputGate = Matrix.sigmoid(Matrix.add(Matrix.add(h0, h1), network["inputBias" + a]));
		
		var h2 = Matrix.multiply(network["forget" + a], previousLayer);
		var h3 = Matrix.multiply(network["forgetHidden" + a], network["outputLast" + a]);
		var forgetGate = Matrix.sigmoid(Matrix.add(Matrix.add(h2, h3), network["forgetBias" + a]));
		
		var h4 = Matrix.multiply(network["output" + a], previousLayer);
		var h5 = Matrix.multiply(network["outputHidden" + a], network["outputLast" + a]);
		var outputGate = Matrix.sigmoid(Matrix.add(Matrix.add(h4, h5), network["outputBias" + a]));
		
		var h6 = Matrix.multiply(network["cell" + a], previousLayer);
		var h7 = Matrix.multiply(network["cellHidden" + a], network["outputLast" + a]);
		var cellWrite = Matrix.hyperbolicTangent(Matrix.add(Matrix.add(h6, h7), network["cellBias" + a]));
		
		var retain = Matrix.elementMultiply(forgetGate, network["cellLast" + a]);
		var write = Matrix.elementMultiply(inputGate, cellWrite);
		
		network["cellLast" + a] = Matrix.add(retain, write);
		network["outputLast" + a] = Matrix.elementMultiply(outputGate, Matrix.hyperbolicTangent(network["cellLast" + a]));
		
	}
	
	network["output"] = Matrix.add(Matrix.multiply(network["decoder"], network["outputLast" + (layerSizes.length - 2)]), network["decoderBias"]);
	
}

examples.initRNN = function () {
	
	var network = this.network;
	
	network["inputLetters"] = new Matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		network["cell" + a] = new Matrix(size, prevSize).randomiseNormalised();
		network["cellHidden" + a] = new Matrix(size, size).randomiseNormalised();
		network["cellBias" + a] = new Matrix(size, 1);
		
		network["outputLast" + a] = new Matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	network["decoder"] = new Matrix(lastLayerSize, layerSizes[layerSizes.length - 2]).randomiseNormalised();
	network["decoderBias"] = new Matrix(lastLayerSize, 1);
	
	network["output"] = new Matrix(lastLayerSize, 1);
	network["desiredValues"] = new Matrix(lastLayerSize, 1);
	
}

examples.updateRNN = function () {
	
	var network = this.network;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? network["inputLetters"] : network["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(network["cell" + a], previousLayer);
		var h1 = Matrix.multiply(network["cellHidden" + a], network["outputLast" + a]);
		var hiddenValues = Matrix.rectifiedLinear(Matrix.add(Matrix.add(h0, h1), network["cellBias" + a]));
		
		network["outputLast" + a] = hiddenValues;
		
	}
	
	network["output"] = Matrix.add(Matrix.multiply(network["decoder"], network["outputLast" + (layerSizes.length - 2)]), network["decoderBias"]);
	
}

examples.initGRU = function () {
	
	var network = this.network;
	
	network["inputLetters"] = new Matrix(layerSizes[0], 1).randomiseNormalised();
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		
		network["update" + a] = new Matrix(layerSizes[a], prevSize).randomiseNormalised();
		network["updateHidden" + a] = new Matrix(layerSizes[a], layerSizes[a]).randomiseNormalised();
		network["updateBias" + a] = new Matrix(layerSizes[a], 1);
		
		network["reset" + a] = new Matrix(layerSizes[a], prevSize).randomiseNormalised();
		network["resetHidden" + a] = new Matrix(layerSizes[a], layerSizes[a]).randomiseNormalised();
		network["resetBias" + a] = new Matrix(layerSizes[a], 1);
		
		network["cell" + a] = new Matrix(layerSizes[a], prevSize).randomiseNormalised();
		network["cellHidden" + a] = new Matrix(layerSizes[a], layerSizes[a]).randomiseNormalised();
		network["cellBias" + a] = new Matrix(layerSizes[a], 1);
		
		network["outputLast" + a] = new Matrix(layerSizes[a], 1);
		
	}
	
	network["decoder"] = new Matrix(layerSizes[layerSizes.length - 1], layerSizes[layerSizes.length - 2]).randomiseNormalised();
	network["decoderBias"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	
	network["output"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	network["desiredValues"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	
}

examples.updateGRU = function () {
	
	var network = this.network;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? network["inputLetters"] : network["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(network["update" + a], previousLayer);
		var h1 = Matrix.multiply(network["updateHidden" + a], network["outputLast" + a]);
		var updateGate = Matrix.sigmoid(Matrix.add(Matrix.add(h0, h1), network["updateBias" + a]));
		
		var h2 = Matrix.multiply(network["reset" + a], previousLayer);
		var h3 = Matrix.multiply(network["resetHidden" + a], network["outputLast" + a]);
		var resetGate = Matrix.sigmoid(Matrix.add(Matrix.add(h2, h3), network["resetBias" + a]));
		
		var h6 = Matrix.multiply(network["cell" + a], Matrix.elementMultiply(previousLayer, resetGate));
		var h7 = Matrix.multiply(network["cellHidden" + a], network["outputLast" + a]);
		var cellWrite = Matrix.hyperbolicTangent(Matrix.add(Matrix.add(h6, h7), network["cellBias" + a]));
		
		network["outputLast" + a] = Matrix.add(Matrix.elementMultiply(Matrix.invert(updateGate), cellWrite), Matrix.elementMultiply(updateGate, network["outputLast" + a]));
		
	}
	
	network["output"] = Matrix.add(Matrix.multiply(network["decoder"], network["outputLast" + (layerSizes.length - 2)]), network["decoderBias"]);
	
}