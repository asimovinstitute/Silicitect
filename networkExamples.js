/*Silicitect, model neural net architectures in JavaScript for silicon hardware.
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
var seed = 3;

// global variables for the examples
var sil = null;
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
	
	Art.doStyle(0, "whiteSpace", "pre-wrap", "font", "20px monospace", "tabSize", "4", "background", "#333", "color", "#ccc");
	
};

function init (e) {
	
	// break out init and update?
	sil = new Silicitect(examples.initLSTM, examples.updateLSTM);
	
	sil.parseText(e.responseText);
	
	layerSizes = [sil.text.characterSet.length, 10, 10, sil.text.characterSet.length];
	// layerSizes = [2, 2, 1];
	// layerSizes = [5, 4, 3, 4, 5];
	
	sil.reguliser = 1e-8;
	sil.learningRate = 0.1;
	sil.clipValue = 5;
	sil.decay = 0.95;
	sil.decayLinear = 0.9;
	sil.optimiser = Silicitect.adamOptimiser;
	
	sil.init();
	
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
		// trainVariationalAutoencoder();
		
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
	// printVariationalAutoencoder();
	
}

Stecy.sequence("update", [doNetworkStuff]);

function trainLogicGate () {
	
	sil.weights[sil.net["input"].i + 0] = +(sil.randomUniform() > 0.5);
	sil.weights[sil.net["input"].i + 1] = +(sil.randomUniform() > 0.5);
	sil.weights[sil.net["desiredValues"].i] = +(sil.weights[sil.net["input"].i + 0] + sil.weights[sil.net["input"].i + 1] > 0);
	
	sil.update();
	sil.computeLoss("output", "desiredValues", "matrixClone", Silicitect.linearLoss);
	sil.backpropagate();
	
}

function printLogicGate () {
	
	sil.weights[sil.net["input"].i + 0] = 0;
	sil.weights[sil.net["input"].i + 1] = 0;
	sil.update();
	Art.doWrite(0, "\n00 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	sil.weights[sil.net["input"].i + 0] = 0;
	sil.weights[sil.net["input"].i + 1] = 1;
	sil.update();
	Art.doWrite(0, "01 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	sil.weights[sil.net["input"].i + 0] = 1;
	sil.weights[sil.net["input"].i + 1] = 0;
	sil.update();
	Art.doWrite(0, "10 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	sil.weights[sil.net["input"].i + 0] = 1;
	sil.weights[sil.net["input"].i + 1] = 1;
	sil.update();
	Art.doWrite(0, "11 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	
}

function trainVariationalAutoencoder () {
	
	
	
}

function printVariationalAutoencoder () {
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		sil.weights[sil.net["input"].i + a] = +(sil.randomUniform() < 0.5);
		
	}
	
	sil.update();
	
	Art.doWrite(0, " ");
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		Art.doWrite(0, sil.weights[sil.net["input"].i + a].toFixed(0));
		
	}
	
	Art.doWrite(0, " ");
	
	for (var a = 0; a < sil.net["output"].l; a++) {
		
		Art.doWrite(0, sil.weights[sil.net["output"].i + a].toFixed(0));
		
	}
	
	Art.doWrite(0, "\n");
	
}

function trainAutoencoder () {
	
	var set = sil.randomUniform() < 0.5 ? [0, 0, 0, 0, 1] : [1, 1, 1, 1, 0];
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		sil.weights[sil.net["input"].i + a] = set[a];
		
	}
	
	sil.update();
	sil.computeLoss("output", "input", "matrixClone", Silicitect.linearLoss);
	sil.backpropagate();
	
}

function printAutoencoder () {
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		sil.weights[sil.net["input"].i + a] = +(sil.randomUniform() < 0.5);
		
	}
	
	sil.update();
	
	Art.doWrite(0, " ");
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		Art.doWrite(0, sil.weights[sil.net["input"].i + a].toFixed(0));
		
	}
	
	Art.doWrite(0, " ");
	
	for (var a = 0; a < sil.net["output"].l; a++) {
		
		Art.doWrite(0, sil.weights[sil.net["output"].i + a].toFixed(0));
		
	}
	
	Art.doWrite(0, "\n");
	
}

function drawAutoencoder () {
	
	Art.canvas.clearRect(0, 0, Art.width, Art.height);
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		sil.net["input"].w[a] = +(sil.randomUniform() < 0.5);
		
	}
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		var value = sil.net["input"].w[a];
		
		Art.canvas.fillStyle = "rgb(255, " + Math.floor(value * 256) + ", 0)";
		Art.canvas.fillRect(a * 100, 0, 100, 50);
		
	}
	
	for (var a = 0; a < sil.net["output"].l; a++) {
		
		var value = sil.net["output"].w[a];
		
		Art.canvas.fillStyle = "rgb(255, " + Math.floor(value * 256) + ", 0)";
		Art.canvas.fillRect(a * 100, 50, 100, 50);
		
	}
	
	sil.update();
	
}

function trainCharacterSequence () {
	
	var sentence = sil.text.raw.substr(Math.floor(sil.randomUniform() * (sil.text.raw.length - sequenceLength)), sequenceLength);
	
	resetLastValues();
	
	for (var a = 0; a < sentence.length - 1; a++) {
		
		sil.matrixFillExcept(sil.net["inputLetters"], 0, sil.text.charToIndex[sentence.charAt(a)], 1);
		sil.matrixFillExcept(sil.net["desiredValues"], 0, sil.text.charToIndex[sentence.charAt(a + 1)], 1);
		sil.update();
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
		sil.update();
		
	}
	
	resetLastValues();
	
	for (var a = 0; a < length; a++) {
		
		var letter = sil.text.charToIndex[sentence.charAt(sentence.length - 1)];
		
		sil.matrixFillExcept(sil.net["inputLetters"], 0, letter, 1);
		
		sil.update();
		
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

examples.initFF = function () {
	
	var net = this.net;
	
	net["input"] = this.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		net["hidden" + a] = this.matrixRandomiseNormalised(this.matrix(size, prevSize));
		net["hiddenBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
		
		net["outputLast" + a] = this.matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = this.matrixRandomiseNormalised(this.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = this.matrixFill(this.matrix(lastLayerSize, 1), 1);
	
	net["output"] = this.matrix(lastLayerSize, 1);
	net["desiredValues"] = this.matrix(lastLayerSize, 1);
	
};

examples.updateFF = function () {
	
	var net = this.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["input"] : net["outputLast" + (a - 1)];
		
		var h0 = this.matrixMultiply(net["hidden" + a], previousLayer);
		
		net["outputLast" + a] = this.matrixSigmoid(this.matrixAdd(h0, net["hiddenBias" + a]));
		
	}
	
	net["output"] = this.matrixSigmoid(this.matrixAdd(this.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]));
	
};

examples.initVAE = function () {
	
	var net = this.net;
	
	net["input"] = this.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		if (a == (layerSizes.length - 1) / 2) {
			
			net["mean"] = this.matrixRandomiseNormalised(this.matrix(size, prevSize));
			net["meanBias"] = this.matrixFill(this.matrix(size, 1), 1);
			
			net["standardDeviation"] = this.matrixRandomiseNormalised(this.matrix(size, prevSize));
			net["standardDeviationBias"] = this.matrixFill(this.matrix(size, 1), 1);
			
		} else {
			
			net["hidden" + a] = this.matrixRandomiseNormalised(this.matrix(size, prevSize));
			net["hiddenBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
			
			net["outputLast" + a] = this.matrix(size, 1);
			
		}
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = this.matrixRandomiseNormalised(this.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = this.matrixFill(this.matrix(lastLayerSize, 1), 1);
	
	net["output"] = this.matrix(lastLayerSize, 1);
	
};

examples.updateVAE = function () {
	
	var net = this.net;
	
	
	
};

examples.initAE = function () {
	
	var net = this.net;
	
	net["input"] = this.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		net["hidden" + a] = this.matrixRandomiseNormalised(this.matrix(size, prevSize));
		net["hiddenBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
		
		net["outputLast" + a] = this.matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = this.matrixRandomiseNormalised(this.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = this.matrixFill(this.matrix(lastLayerSize, 1), 1);
	
	net["output"] = this.matrix(lastLayerSize, 1);
	
};

examples.updateAE = function () {
	
	var net = this.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["input"] : net["outputLast" + (a - 1)];
		
		var h0 = this.matrixMultiply(net["hidden" + a], previousLayer);
		
		net["outputLast" + a] = this.matrixSigmoid(this.matrixAdd(h0, net["hiddenBias" + a]));
		
	}
	
	net["output"] = this.matrixSigmoid(this.matrixAdd(this.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]));
	
};

examples.initLSTM = function () {
	
	var net = this.net;
	
	net["inputLetters"] = this.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		net["input" + a] = this.matrixRandomiseGaussian(this.matrix(size, prevSize), 0, 2);
		net["inputHidden" + a] = this.matrixRandomiseGaussian(this.matrix(size, size), 0, 2);
		net["inputBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
		
		net["forget" + a] = this.matrixRandomiseGaussian(this.matrix(size, prevSize), 0, 2);
		net["forgetHidden" + a] = this.matrixRandomiseGaussian(this.matrix(size, size), 0, 2);
		net["forgetBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
		
		net["output" + a] = this.matrixRandomiseGaussian(this.matrix(size, prevSize), 0, 2);
		net["outputHidden" + a] = this.matrixRandomiseGaussian(this.matrix(size, size), 0, 2);
		net["outputBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
		net["outputLast" + a] = this.matrix(size, 1);
		
		net["cell" + a] = this.matrixRandomiseGaussian(this.matrix(size, prevSize), 0, 2);
		net["cellHidden" + a] = this.matrixRandomiseGaussian(this.matrix(size, size), 0, 2);
		net["cellBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
		net["cellLast" + a] = this.matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = this.matrixRandomiseGaussian(this.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]), 0, 2);
	net["decoderBias"] = this.matrixFill(this.matrix(lastLayerSize, 1), 1);
	
	net["output"] = this.matrix(lastLayerSize, 1);
	net["desiredValues"] = this.matrix(lastLayerSize, 1);
	
};

examples.updateLSTM = function () {
	
	var net = this.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["inputLetters"] : net["outputLast" + (a - 1)];
		
		var h0 = this.matrixMultiply(net["input" + a], previousLayer);
		var h1 = this.matrixMultiply(net["inputHidden" + a], net["outputLast" + a]);
		var inputGate = this.matrixSigmoid(this.matrixAdd(this.matrixAdd(h0, h1), net["inputBias" + a]));
		
		var h2 = this.matrixMultiply(net["forget" + a], previousLayer);
		var h3 = this.matrixMultiply(net["forgetHidden" + a], net["outputLast" + a]);
		var forgetGate = this.matrixSigmoid(this.matrixAdd(this.matrixAdd(h2, h3), net["forgetBias" + a]));
		
		var h4 = this.matrixMultiply(net["output" + a], previousLayer);
		var h5 = this.matrixMultiply(net["outputHidden" + a], net["outputLast" + a]);
		var outputGate = this.matrixSigmoid(this.matrixAdd(this.matrixAdd(h4, h5), net["outputBias" + a]));
		
		var h6 = this.matrixMultiply(net["cell" + a], previousLayer);
		var h7 = this.matrixMultiply(net["cellHidden" + a], net["outputLast" + a]);
		var cellWrite = this.matrixHyperbolicTangent(this.matrixAdd(this.matrixAdd(h6, h7), net["cellBias" + a]));
		
		var retain = this.matrixElementMultiply(forgetGate, net["cellLast" + a]);
		var write = this.matrixElementMultiply(inputGate, cellWrite);
		
		net["cellLast" + a] = this.matrixAdd(retain, write);
		net["outputLast" + a] = this.matrixElementMultiply(outputGate, this.matrixHyperbolicTangent(net["cellLast" + a]));
		
	}
	
	net["output"] = this.matrixAdd(this.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]);
	
};

examples.initRNN = function () {
	
	var net = this.net;
	
	net["inputLetters"] = this.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		net["cell" + a] = this.matrixRandomiseNormalised(this.matrix(size, prevSize));
		net["cellHidden" + a] = this.matrixRandomiseNormalised(this.matrix(size, size));
		net["cellBias" + a] = this.matrixFill(this.matrix(size, 1), 1);
		
		net["outputLast" + a] = this.matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = this.matrixRandomiseNormalised(this.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = this.matrixFill(this.matrix(lastLayerSize, 1), 1);
	
	net["output"] = this.matrix(lastLayerSize, 1);
	net["desiredValues"] = this.matrix(lastLayerSize, 1);
	
};

examples.updateRNN = function () {
	
	var net = this.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["inputLetters"] : net["outputLast" + (a - 1)];
		
		var h0 = this.matrixMultiply(net["cell" + a], previousLayer);
		var h1 = this.matrixMultiply(net["cellHidden" + a], net["outputLast" + a]);
		var hiddenValues = this.matrixRectifiedLinear(this.matrixAdd(this.matrixAdd(h0, h1), net["cellBias" + a]));
		
		net["outputLast" + a] = hiddenValues;
		
	}
	
	net["output"] = this.matrixAdd(this.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]);
	
	
};

examples.initGRU = function () {
	
	var net = this.net;
	
	net["inputLetters"] = this.matrixRandomiseUniform(this.matrix(layerSizes[0], 1));
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		
		net["update" + a] = this.matrixRandomiseNormalised(this.matrix(layerSizes[a], prevSize));
		net["updateHidden" + a] = this.matrixRandomiseNormalised(this.matrix(layerSizes[a], layerSizes[a]));
		net["updateBias" + a] = this.matrix(layerSizes[a], 1);
		
		net["reset" + a] = this.matrixRandomiseNormalised(this.matrix(layerSizes[a], prevSize));
		net["resetHidden" + a] = this.matrixRandomiseNormalised(this.matrix(layerSizes[a], layerSizes[a]));
		net["resetBias" + a] = this.matrix(layerSizes[a], 1);
		
		net["cell" + a] = this.matrixRandomiseNormalised(this.matrix(layerSizes[a], prevSize));
		net["cellHidden" + a] = this.matrixRandomiseNormalised(this.matrix(layerSizes[a], layerSizes[a]));
		net["cellBias" + a] = this.matrix(layerSizes[a], 1);
		
		net["outputLast" + a] = this.matrix(layerSizes[a], 1);
		
	}
	
	net["decoder"] = this.matrixRandomiseNormalised(this.matrix(layerSizes[layerSizes.length - 1], layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = this.matrix(layerSizes[layerSizes.length - 1], 1);
	
	net["output"] = this.matrix(layerSizes[layerSizes.length - 1], 1);
	net["desiredValues"] = this.matrix(layerSizes[layerSizes.length - 1], 1);
	
};

examples.updateGRU = function () {
	
	var net = this.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["inputLetters"] : net["outputLast" + (a - 1)];
		
		var h0 = this.matrixMultiply(net["update" + a], previousLayer);
		var h1 = this.matrixMultiply(net["updateHidden" + a], net["outputLast" + a]);
		var updateGate = this.matrixSigmoid(this.matrixAdd(this.matrixAdd(h0, h1), net["updateBias" + a]));
		
		var h2 = this.matrixMultiply(net["reset" + a], previousLayer);
		var h3 = this.matrixMultiply(net["resetHidden" + a], net["outputLast" + a]);
		var resetGate = this.matrixSigmoid(this.matrixAdd(this.matrixAdd(h2, h3), net["resetBias" + a]));
		
		var h6 = this.matrixMultiply(net["cell" + a], this.matrixElementMultiply(previousLayer, resetGate));
		var h7 = this.matrixMultiply(net["cellHidden" + a], net["outputLast" + a]);
		var cellWrite = this.matrixHyperbolicTangent(this.matrixAdd(this.matrixAdd(h6, h7), net["cellBias" + a]));
		
		net["outputLast" + a] = this.matrixAdd(this.matrixElementMultiply(this.matrixInvert(updateGate), cellWrite), this.matrixElementMultiply(updateGate, net["outputLast" + a]));
		
	}
	
	net["output"] = this.matrixAdd(this.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]);
	
};