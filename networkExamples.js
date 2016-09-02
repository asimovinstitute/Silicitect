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

var running = false;
var iterationsPerFrame = 100;
var maxIterations = 2e3;
var letterCount = 10;
var sampleSize = 50;
var averageBatchTime = 0;
var filePath = "input/simple.txt";
var samplePrime = "0";
var totalIterations = 0;
var layerSizes = [];

var sil = null;
var textParser = null;

Stecy.sequence("update", [doNetworkStuff]);

Stecy.setup = function () {
	
	Art.title = "Silicitect";
	
};

Art.ready = function () {
	
	Stecy.loadFile(filePath, init);
	
	Art.doStyle(0, "whiteSpace", "pre", "font", "20px monospace", "tabSize", "6", "background", "#333", "color", "#ccc");
	
};

function init (e) {
	
	Art.doStyle(0, "whiteSpace", "pre-wrap");
	
	// textParser = new TextParser(e.responseText, );
	textParser = new TextParser(e.responseText, "");
	layerSizes = [textParser.chars.length, 10, 10, textParser.chars.length];
	// layerSizes = [2, 2, 1];
	
	sil = new Silicitect(initLSTM, updateLSTM);
	sil.reguliser = 1e-8;
	sil.learningRate = 0.1;
	sil.clipValue = 5;
	sil.decayRate = 0.95;
	
	Matrix.silicitect = sil;
	
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
	Art.doWrite(0, totalIterations + " " + (sil.totalLoss / iterationsPerFrame).toFixed(2) + " " + sil.batchTime + "ms" +
					" avg " + (averageBatchTime / (totalIterations / iterationsPerFrame)).toFixed(2));
	
	printCharacterSequenceOutput();
	
}

function trainLogicGate () {
	
	sil.model["input"].w = [uniform() > 0.5, uniform() > 0.5];
	sil.model["desiredValues"].w = [sil.model["input"].w[0] + sil.model["input"].w[1] > 0];
	sil.update();
	sil.computeLoss("output", "desiredValues", Matrix.nothing, Silicitect.linearLoss);
	sil.backpropagate();
	
}

function printLogicGateOutput () {
	
	sil.model["input"].w = [0, 0];
	sil.update();
	Art.doWrite(0, "\n00 " + sil.model["output"].w[0].toFixed(2) + "\n");
	sil.model["input"].w = [0, 1];
	sil.update();
	Art.doWrite(0, "01 " + sil.model["output"].w[0].toFixed(2) + "\n");
	sil.model["input"].w = [1, 0];
	sil.update();
	Art.doWrite(0, "10 " + sil.model["output"].w[0].toFixed(2) + "\n");
	sil.model["input"].w = [1, 1];
	sil.update();
	Art.doWrite(0, "11 " + sil.model["output"].w[0].toFixed(2) + "\n");
	
}

function trainCharacterSequence () {
	
	var sentence = textParser.text.substr(Math.floor(uniform() * (textParser.text.length - letterCount)), letterCount);
	
	resetLastValues();
	
	for (var a = 0; a < sentence.length - 1; a++) {
		
		sil.model["inputLetters"].fillZerosExcept(textParser.charToIndex[sentence.charAt(a)]);
		sil.model["desiredValues"].fillZerosExcept(textParser.charToIndex[sentence.charAt(a + 1)]);
		sil.update();
		sil.computeLoss("output", "desiredValues", Matrix.softmax, Silicitect.logLoss);
		
	}
	
	sil.backpropagate();
	
}

function printCharacterSequenceOutput () {
	
	Art.doWrite(0, " " + generateSentence(sampleSize, samplePrime) + "\n");
	
}

function generateSentence (length, prime) {
	
	var sentence = prime;
	
	resetLastValues();
	
	for (var a = 0; a < prime.length; a++) {
		
		var letter = textParser.charToIndex[prime.charAt(a)];
		
		if (!(1 + letter)) continue;
		
		sil.model["inputLetters"].fillZerosExcept(letter);
		sil.update();
		
	}
	
	resetLastValues();
	
	for (var a = 0; a < length; a++) {
		
		var letter = textParser.charToIndex[sentence.charAt(sentence.length - 1)];
		
		sil.model["inputLetters"].fillZerosExcept(letter);
		
		sil.update();
		
		var probabilities = Matrix.softmax(sil.model["output"], 1.0);
		var index = Matrix.sampleRandomSum(probabilities);
		
		sentence += textParser.chars.charAt(index);
		
	}
	
	return sentence.slice(prime.length);
	
}

function resetLastValues () {
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		sil.model["outputLast" + a].fillZeros();
		sil.model["cellLast" + a].fillZeros();
		
	}
	
}

function updateFF (model) {
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? model["input"] : model["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(model["hidden" + a], previousLayer);
		
		model["outputLast" + a] = Matrix.sigmoid(Matrix.add(h0, model["hiddenBias" + a]));
		
	}
	
	model["output"] = Matrix.sigmoid(Matrix.add(Matrix.multiply(model["decoder"], model["outputLast" + (layerSizes.length - 2)]), model["decoderBias"]));
	
}

function initFF (model) {
	
	model["input"] = new Matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		model["hidden" + a] = new Matrix(size, prevSize).randomiseNormalised();
		model["hiddenBias" + a] = new Matrix(size, 1).fillOnes();
		
		model["outputLast" + a] = new Matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	model["decoder"] = new Matrix(lastLayerSize, layerSizes[layerSizes.length - 2]).randomiseNormalised();
	model["decoderBias"] = new Matrix(lastLayerSize, 1).fillOnes();
	
	model["output"] = new Matrix(lastLayerSize, 1);
	model["desiredValues"] = new Matrix(lastLayerSize, 1);
	
}

function updateLSTM (model) {
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? model["inputLetters"] : model["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(model["input" + a], previousLayer);
		var h1 = Matrix.multiply(model["inputHidden" + a], model["outputLast" + a]);
		var inputGate = Matrix.sigmoid(Matrix.add(Matrix.add(h0, h1), model["inputBias" + a]));
		
		var h2 = Matrix.multiply(model["forget" + a], previousLayer);
		var h3 = Matrix.multiply(model["forgetHidden" + a], model["outputLast" + a]);
		var forgetGate = Matrix.sigmoid(Matrix.add(Matrix.add(h2, h3), model["forgetBias" + a]));
		
		var h4 = Matrix.multiply(model["output" + a], previousLayer);
		var h5 = Matrix.multiply(model["outputHidden" + a], model["outputLast" + a]);
		var outputGate = Matrix.sigmoid(Matrix.add(Matrix.add(h4, h5), model["outputBias" + a]));
		
		var h6 = Matrix.multiply(model["cell" + a], previousLayer);
		var h7 = Matrix.multiply(model["cellHidden" + a], model["outputLast" + a]);
		var cellWrite = Matrix.hyperbolicTangent(Matrix.add(Matrix.add(h6, h7), model["cellBias" + a]));
		
		var retain = Matrix.elementMultiply(forgetGate, model["cellLast" + a]);
		var write = Matrix.elementMultiply(inputGate, cellWrite);
		
		model["cellLast" + a] = Matrix.add(retain, write);
		model["outputLast" + a] = Matrix.elementMultiply(outputGate, Matrix.hyperbolicTangent(model["cellLast" + a]));
		
	}
	
	model["output"] = Matrix.add(Matrix.multiply(model["decoder"], model["outputLast" + (layerSizes.length - 2)]), model["decoderBias"]);
	
}

function initLSTM (model) {
	
	model["inputLetters"] = new Matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		model["input" + a] = new Matrix(size, prevSize).randomiseNormalised();
		model["inputHidden" + a] = new Matrix(size, size).randomiseNormalised();
		model["inputBias" + a] = new Matrix(size, 1).fillOnes();
		
		model["forget" + a] = new Matrix(size, prevSize).randomiseNormalised();
		model["forgetHidden" + a] = new Matrix(size, size).randomiseNormalised();
		model["forgetBias" + a] = new Matrix(size, 1).fillOnes();
		
		model["output" + a] = new Matrix(size, prevSize).randomiseNormalised();
		model["outputHidden" + a] = new Matrix(size, size).randomiseNormalised();
		model["outputBias" + a] = new Matrix(size, 1).fillOnes();
		model["outputLast" + a] = new Matrix(size, 1);
		
		model["cell" + a] = new Matrix(size, prevSize).randomiseNormalised();
		model["cellHidden" + a] = new Matrix(size, size).randomiseNormalised();
		model["cellBias" + a] = new Matrix(size, 1).fillOnes();
		model["cellLast" + a] = new Matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	model["decoder"] = new Matrix(lastLayerSize, layerSizes[layerSizes.length - 2]).randomiseNormalised();
	model["decoderBias"] = new Matrix(lastLayerSize, 1).fillOnes();
	
	model["output"] = new Matrix(lastLayerSize, 1);
	model["desiredValues"] = new Matrix(lastLayerSize, 1);
	
}

function initRNN () {
	
	model = {};
	
	model["inputLetters"] = new Matrix(layerSizes[0], 1).randomiseNormalised();
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		
		model["cell" + a] = new Matrix(layerSizes[a], prevSize).randomiseNormalised();
		model["cellHidden" + a] = new Matrix(layerSizes[a], layerSizes[a]).randomiseNormalised();
		model["cellBias" + a] = new Matrix(layerSizes[a], 1);
		
		model["outputLast" + a] = new Matrix(layerSizes[a], 1);
		
	}
	
	model["decoder"] = new Matrix(layerSizes[layerSizes.length - 1], layerSizes[layerSizes.length - 2]).randomiseNormalised();
	model["decoderBias"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	model["output"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	model["desiredValues"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	
}

function updateRNN (input, firstPass) {
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? Matrix.rowPluck(model["inputLetters"], input) : model["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(model["cell" + a], previousLayer);
		var h1 = Matrix.multiply(model["cellHidden" + a], model["outputLast" + a]);
		var hiddenValues = Matrix.rectifiedLinear(Matrix.add(Matrix.add(h0, h1), model["cellBias" + a]));
		
		model["outputLast" + a] = hiddenValues;
		
	}
	
	model["output"] = Matrix.add(Matrix.multiply(model["decoder"], model["outputLast" + (layerSizes.length - 2)]), model["decoderBias"]);
	
}

function initGRU () {
	
	model = {};
	
	model["inputLetters"] = new Matrix(layerSizes[0], 1).randomiseNormalised();
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		
		model["update" + a] = new Matrix(layerSizes[a], prevSize).randomiseNormalised();
		model["updateHidden" + a] = new Matrix(layerSizes[a], layerSizes[a]).randomiseNormalised();
		model["updateBias" + a] = new Matrix(layerSizes[a], 1);
		
		model["reset" + a] = new Matrix(layerSizes[a], prevSize).randomiseNormalised();
		model["resetHidden" + a] = new Matrix(layerSizes[a], layerSizes[a]).randomiseNormalised();
		model["resetBias" + a] = new Matrix(layerSizes[a], 1);
		
		model["cell" + a] = new Matrix(layerSizes[a], prevSize).randomiseNormalised();
		model["cellHidden" + a] = new Matrix(layerSizes[a], layerSizes[a]).randomiseNormalised();
		model["cellBias" + a] = new Matrix(layerSizes[a], 1);
		
		model["outputLast" + a] = new Matrix(layerSizes[a], 1);
		
	}
	
	model["decoder"] = new Matrix(layerSizes[layerSizes.length - 1], layerSizes[layerSizes.length - 2]).randomiseNormalised();
	model["decoderBias"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	model["output"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	model["desiredValues"] = new Matrix(layerSizes[layerSizes.length - 1], 1);
	
}

function updateGRU (input, firstPass) {
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? Matrix.rowPluck(model["inputLetters"], input) : model["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(model["update" + a], previousLayer);
		var h1 = Matrix.multiply(model["updateHidden" + a], model["outputLast" + a]);
		var updateGate = Matrix.sigmoid(Matrix.add(Matrix.add(h0, h1), model["updateBias" + a]));
		
		var h2 = Matrix.multiply(model["reset" + a], previousLayer);
		var h3 = Matrix.multiply(model["resetHidden" + a], model["outputLast" + a]);
		var resetGate = Matrix.sigmoid(Matrix.add(Matrix.add(h2, h3), model["resetBias" + a]));
		
		var h6 = Matrix.multiply(model["cell" + a], Matrix.elementMultiply(previousLayer, resetGate));
		var h7 = Matrix.multiply(model["cellHidden" + a], model["outputLast" + a]);
		var cellWrite = Matrix.hyperbolicTangent(Matrix.add(Matrix.add(h6, h7), model["cellBias" + a]));
		
		model["outputLast" + a] = Matrix.add(Matrix.elementMultiply(Matrix.invert(updateGate), cellWrite), Matrix.elementMultiply(updateGate, model["outputLast" + a]));
		
	}
	
	model["output"] = Matrix.add(Matrix.multiply(model["decoder"], model["outputLast" + (layerSizes.length - 2)]), model["decoderBias"]);
	
}

var rgn = 0;
var seed = 3;

function uniform () {
	
	rgn = (4321421413 * rgn + 432194612 + seed) % 43214241 * (79143569 + seed);
	
	return 1e-10 * (rgn % 1e10);
	
}