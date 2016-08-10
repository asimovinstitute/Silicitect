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

var reguliser = 1e-8;
var learningRate = 0.2;
var clipValue = 5;
var decayRate = 0.97;

var layers = [];
var model = {};
var lastWeights = {};
var recordBackprop = false;
var backprop = [];

var letterEmbedSize = 5;
var letterCount = 10;

var running = false;
var iterationsPerFrame = 500;
var maxIterations = 1000;
var sampleSize = 20;
var samplePrime = "0";
var networkType = "lstm";

var totalIterations = 0;
var textParser;

function init (e) {
	
	// textParser = new TextParser(e.responseText, "!@#$%^&*()_+{}\":|?><~±§¡€£¢∞œŒ∑´®†¥øØπ∏¬˚∆åÅßΩéúíóáÉÚÍÓÁëüïöäËÜÏÖÄ™‹›ﬁﬂ‡°·—≈çÇ√-=[];',.\\/`µ≤≥„‰◊ˆ˜¯˘¿⁄\n\t" + 
	// 			"1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
	textParser = new TextParser(e.responseText, "");
	
	layers = [textParser.characters.length, 10, 10, textParser.characters.length];
	
	if (networkType == "lstm") initLSTM();
	else initRNN();
	
	running = true;
	
}

function doNetworkStuff () {
	
	if (!running) return;
	
	if (totalIterations >= maxIterations) return;
	
	var startTime = new Date();
	var averageLoss = 0;
	var sentence = "";
	
	for (var a = 0; a < iterationsPerFrame; a++) {
		
		sentence = textParser.text.substr(Math.floor(uniform() * (textParser.text.length - letterCount)), letterCount);
		
		averageLoss += answer(sentence, networkType == "lstm" ? forwardLSTM : forwardRNN);
		
	}
	
	totalIterations += iterationsPerFrame;
	
	console.log(totalIterations,
				(averageLoss / iterationsPerFrame).toFixed(2),
				(new Date() - startTime) + "ms",
				ask(sampleSize, samplePrime, networkType == "lstm" ? forwardLSTM : forwardRNN));
	
}

Stecy.sequence("update", [doNetworkStuff]);

function ask (length, prime, forwardFunction) {
	
	recordBackprop = false;
	
	var sentence = prime;
	var out = {};
	
	for (var a = 0; a < prime.length; a++) {
		
		var letter = textParser.letterToIndex[prime.charAt(a)];
		
		if (!(letter + 1)) continue;
		
		forwardFunction(letter, a == 0);
		
	}
	
	for (var a = 0; a < length; a++) {
		
		var inputLetter = textParser.letterToIndex[sentence.charAt(sentence.length - 1)];
		
		out = forwardFunction(inputLetter, a == 0);
		
		var temperature = 1;
		var probabilities = Matrix.softmax(out, temperature);
		var index = Matrix.sampleMax(probabilities);
		
		sentence += textParser.characters.charAt(index);
		
	}
	
	return sentence.slice(prime.length);
	
}

function answer (sentence, forwardFunction) {
	
	recordBackprop = true;
	backprop = [];
	
	var loss = 0;
	var out = {};
	
	for (var a = 0; a < sentence.length - 1; a++) {
		
		var letter = textParser.letterToIndex[sentence.charAt(a)];
		var nextLetter = textParser.letterToIndex[sentence.charAt(a + 1)];
		
		out = forwardFunction(letter, a == 0);
		
		var probabilities = Matrix.softmax(out);
		
		loss -= Math.log(probabilities.w[nextLetter]);
		
		out.dw = probabilities.w;
		out.dw[nextLetter] -= 1;
		
	}
	
	backward();
	
	return loss;
	
}

function backward () {
	
	for (var a = backprop.length - 1; a > -1; a -= 2) {
		
		if (backprop[a].length == 1) backprop[a - 1](backprop[a][0]);
		if (backprop[a].length == 2) backprop[a - 1](backprop[a][0], backprop[a][1]);
		if (backprop[a].length == 3) backprop[a - 1](backprop[a][0], backprop[a][1], backprop[a][2]);
		
	}
	
	for (var a in model) {
		
		if (!lastWeights[a]) lastWeights[a] = new Matrix(model[a].n, model[a].d);
		
		var ma = model[a];
		var mb = lastWeights[a];
		
		for (var b = 0; b < ma.w.length; b++) {
			
			mb.w[b] = mb.w[b] * decayRate + (1 - decayRate) * ma.dw[b] * ma.dw[b];
			
			var clippedValue = Math.max(-clipValue, Math.min(clipValue, ma.dw[b]));
			
			ma.w[b] += -learningRate * clippedValue / Math.sqrt(mb.w[b] + 1e-8) - reguliser * ma.w[b];
			ma.dw[b] = 0;
			
		}
		
	}
	
}

function forwardRNN (input, firstPass) {
	
	if (firstPass) {
		
		for (var a = 1; a < layers.length - 1; a++) {
			
			model["outputLast" + a] = new Matrix(layers[a], 1);
			
		}
		
	}
	
	for (var a = 1; a < layers.length - 1; a++) {
		
		var previousLayer = a == 1 ? Matrix.rowPluck(model["inputLetters"], input) : model["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(model["weight" + a], previousLayer);
		var h1 = Matrix.multiply(model["weightRecurrent" + a], model["outputLast" + a]);
		var hiddenValues = Matrix.rectifiedLinearUnit(Matrix.add(Matrix.add(h0, h1), model["weightBias" + a]));
		
		model["outputLast" + a] = hiddenValues;
		
	}
	
	var output = Matrix.add(Matrix.multiply(model["decoder"], model["outputLast" + (layers.length - 2)]), model["decoderBias"]);
	
	return output;
	
}

function forwardLSTM (input, firstPass) {
	
	if (firstPass) {
		
		for (var a = 1; a < layers.length - 1; a++) {
			
			model["outputLast" + a] = new Matrix(layers[a], 1);
			model["cellLast" + a] = new Matrix(layers[a], 1);
			
		}
		
	}
	
	for (var a = 1; a < layers.length - 1; a++) {
		
		var previousLayer = a == 1 ? Matrix.rowPluck(model["inputLetters"], input) : model["outputLast" + (a - 1)];
		
		var h0 = Matrix.multiply(model["input" + a], previousLayer);
		var h1 = Matrix.multiply(model["inputRecurrent" + a], model["outputLast" + a]);
		var inputGate = Matrix.sigmoid(Matrix.add(Matrix.add(h0, h1), model["inputBias" + a]));
		
		var h2 = Matrix.multiply(model["forget" + a], previousLayer);
		var h3 = Matrix.multiply(model["forgetRecurrent" + a], model["outputLast" + a]);
		var forgetGate = Matrix.sigmoid(Matrix.add(Matrix.add(h2, h3), model["forgetBias" + a]));
		
		var h4 = Matrix.multiply(model["output" + a], previousLayer);
		var h5 = Matrix.multiply(model["outputRecurrent" + a], model["outputLast" + a]);
		var outputGate = Matrix.sigmoid(Matrix.add(Matrix.add(h4, h5), model["outputBias" + a]));
		
		var h6 = Matrix.multiply(model["cell" + a], previousLayer);
		var h7 = Matrix.multiply(model["cellRecurrent" + a], model["outputLast" + a]);
		var cellWrite = Matrix.hyperbolicTangent(Matrix.add(Matrix.add(h6, h7), model["cellBias" + a]));
		
		var retain = Matrix.elementMultiply(forgetGate, model["cellLast" + a]);
		var write = Matrix.elementMultiply(inputGate, cellWrite);
		
		model["cellLast" + a] = Matrix.add(retain, write);
		model["outputLast" + a] = Matrix.elementMultiply(outputGate, Matrix.hyperbolicTangent(model["cellLast" + a]));
		
	}
	
	var output = Matrix.add(Matrix.multiply(model["decoder"], model["outputLast" + (layers.length - 2)]), model["decoderBias"]);
	
	return output;
	
}

function initRNN () {
	
	model = {};
	
	model["inputLetters"] = new Matrix(layers[0], letterEmbedSize).randomiseNormalised();
	
	for (var a = 1; a < layers.length - 1; a++) {
		
		var prevSize = a == 1 ? letterEmbedSize : layers[a - 1];
		
		model["weight" + a] = new Matrix(layers[a], prevSize).randomiseNormalised();
		model["weightRecurrent" + a] = new Matrix(layers[a], layers[a]).randomiseNormalised();
		model["weightBias" + a] = new Matrix(layers[a], 1);
		
		model["outputLast" + a] = new Matrix(layers[a], 1);
		
	}
	
	model["decoder"] = new Matrix(layers[layers.length - 1], layers[layers.length - 2]).randomiseNormalised();
	model["decoderBias"] = new Matrix(layers[layers.length - 1], 1);
	
}

function initGRU () {
	
	
	
}

function initLSTM () {
	
	model = {};
	
	model["inputLetters"] = new Matrix(layers[0], letterEmbedSize).randomiseNormalised();
	
	for (var a = 1; a < layers.length - 1; a++) {
		
		var prevSize = a == 1 ? letterEmbedSize : layers[a - 1];
		
		model["input" + a] = new Matrix(layers[a], prevSize).randomiseNormalised();
		model["inputRecurrent" + a] = new Matrix(layers[a], layers[a]).randomiseNormalised();
		model["inputBias" + a] = new Matrix(layers[a], 1);
		
		model["forget" + a] = new Matrix(layers[a], prevSize).randomiseNormalised();
		model["forgetRecurrent" + a] = new Matrix(layers[a], layers[a]).randomiseNormalised();
		model["forgetBias" + a] = new Matrix(layers[a], 1);
		
		model["output" + a] = new Matrix(layers[a], prevSize).randomiseNormalised();
		model["outputRecurrent" + a] = new Matrix(layers[a], layers[a]).randomiseNormalised();
		model["outputBias" + a] = new Matrix(layers[a], 1);
		model["outputLast" + a] = new Matrix(layers[a], 1);
		
		model["cell" + a] = new Matrix(layers[a], prevSize).randomiseNormalised();
		model["cellRecurrent" + a] = new Matrix(layers[a], layers[a]).randomiseNormalised();
		model["cellBias" + a] = new Matrix(layers[a], 1);
		model["cellLast" + a] = new Matrix(layers[a], 1);
		
	}
	
	model["decoder"] = new Matrix(layers[layers.length - 1], layers[layers.length - 2]).randomiseUniform();
	model["decoderBias"] = new Matrix(layers[layers.length - 1], 1);
	
}

Stecy.setup = function () {
	
	Art.title = "Silicitect";
	
};

Art.ready = function () {
	
	Stecy.loadFile("input/simple.txt", init);
	
	Art.doStyle(0, "whiteSpace", "pre", "font", "20px monospace", "tabSize", "6", "background", "#333", "color", "#ccc");
	
};

(function () {
	
	TextParser = function (text, characterSet) {
		
		this.text = text;
		this.letterToIndex = {};
		this.characters = characterSet;
		
		for (var a = 0; a < this.characters.length; a++) {
			
			this.letterToIndex[this.characters.charAt(a)] = a;
			
		}
		
		for (var a = 0; a < this.text.length; a++) {
			
			var char = this.text.charAt(a);
			
			if (1 + this.letterToIndex[char]) continue;
			
			this.letterToIndex[char] = this.characters.length;
			this.characters += char;
			
		}
		
	};
	
	Matrix = function (n, d) {
		
		this.n = n;
		this.d = d;
		this.w = [];
		this.dw = [];
		
		for (var a = 0; a < n * d; a++) {
			
			this.w[a] = 0;
			this.dw[a] = 0;
			
		}
		
	};
	
	Matrix.prototype.randomiseUniform = function () {
		
		for (var a = 0; a < this.n * this.d; a++) {
			
			this.w[a] = uniform();
			
		}
		
		return this;
		
	};
	
	Matrix.prototype.randomiseNormalised = function (base, range) {
		
		for (var a = 0; a < this.n * this.d; a++) {
			
			this.w[a] = uniform() / Math.sqrt(this.d);
			
		}
		
		return this;
		
	};
	
	Matrix.softmax = function (ma, temp) {
		
		var out = new Matrix(ma.n, ma.d);
		var max = -1e10;
		var sum = 0;
		
		if (temp) {
			
			for (var a = 0; a < ma.w.length; a++) ma.w[a] /= temp;
			
		}
		
		for (var a = 0; a < ma.w.length; a++) {
			
			if (ma.w[a] > max) max = ma.w[a];
			
		}
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = Math.exp(ma.w[a] - max);
			
			sum += out.w[a];
			
		}
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] /= sum;
			
		}
		
		return out;
		
	};
	
	Matrix.sampleMax = function (ma) {
		
		var highest = 0;
		
		for (var a = 1; a < ma.w.length; a++) {
			
			if (ma.w[a] > ma.w[highest]) {
				
				highest = a;
				
			}
			
		}
		
		return highest;
		
	};
	
	Matrix.sampleRandomSum = function (ma) {
		
		var random = Math.random();
		var sum = 0;
		
		for (var a = 0; a < ma.w.length; a++) {
			
			sum += ma.w[a];
			
			if (sum > random) return a;
			
		}
		
		return a - 1;
		
	};
	
	Matrix.multiply = function (ma, mb) {
		
		if (ma.d != mb.n) throw new Error("wrong dimensions");
		
		var out = new Matrix(ma.n, mb.d);
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				out.w[mb.d * a + b] = 0;
				
				for (var c = 0; c < ma.d; c++) {
					
					out.w[mb.d * a + b] += ma.w[ma.d * a + c] * mb.w[mb.d * c + b];
					
				}
				
			}
			
		}
		
		if (recordBackprop) backprop.push(Matrix.multiplyBackward, [ma, mb, out]);
		
		return out;
		
	};
	
	Matrix.multiplyBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				for (var c = 0; c < ma.d; c++) {
					
					ma.dw[ma.d * a + c] += mb.w[mb.d * c + b] * out.dw[mb.d * a + b];
					mb.dw[mb.d * c + b] += ma.w[ma.d * a + c] * out.dw[mb.d * a + b];
					
				}
				
			}
			
		}
		
	};
	
	Matrix.elementMultiply = function (ma, mb) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = ma.w[a] * mb.w[a];
			
		}
		
		if (recordBackprop) backprop.push(Matrix.elementMultiplyBackward, [ma, mb, out]);
		
		return out;
		
	};
	
	Matrix.elementMultiplyBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.dw[a] += mb.w[a] * out.dw[a];
			mb.dw[a] += ma.w[a] * out.dw[a];
			
		}
		
	};
	
	Matrix.add = function (ma, mb) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = ma.w[a] + mb.w[a];
			
		}
		
		if (recordBackprop) backprop.push(Matrix.addBackward, [ma, mb, out]);
		
		return out;
		
	};
	
	Matrix.addBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.dw[a] += out.dw[a];
			mb.dw[a] += out.dw[a];
			
		}
		
	};
	
	Matrix.sigmoid = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = 1 / (1 + Math.exp(-ma.w[a]));
			
		}
		
		if (recordBackprop) backprop.push(Matrix.sigmoidBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.sigmoidBackward = function (ma, out) {
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.dw[a] += out.w[a] * (1 - out.w[a]) * out.dw[a];
			
		}
		
	};
	
	Matrix.rectifiedLinearUnit = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = Math.max(0, ma.w[a]);
			
		}
		
		if (recordBackprop) backprop.push(Matrix.rectifiedLinearUnitBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.rectifiedLinearUnitBackward = function (ma, out) {
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.dw[a] += ma.w[a] > 0 ? out.dw[a] : 0;
			
		}
		
	};
	
	Matrix.hyperbolicTangent = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = Math.tanh(ma.w[a]);
			
		}
		
		if (recordBackprop) backprop.push(Matrix.hyperbolicTangentBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.hyperbolicTangentBackward = function (ma, out) {
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.dw[a] += (1 - out.w[a] * out.w[a]) * out.dw[a];
			
		}
		
	};
	
	Matrix.rowPluck = function (ma, row) {
		
		var out = new Matrix(ma.d, 1);
		
		for (var a = 0; a < ma.d; a++) {
			
			out.w[a] = ma.w[ma.d * row + a];
			
		}
		
		if (recordBackprop) backprop.push(Matrix.rowPluckBackward, [ma, out, row]);
		
		return out;
		
	};
	
	Matrix.rowPluckBackward = function (ma, out, row) {
		
		for (var a = 0; a < ma.d; a++) {
			
			ma.dw[ma.d * row + a] += out.dw[a];
			
		}
		
	}
	
})();

var rgn = 0;
var seed = 3;

function uniform () {
	
	rgn = (4321421413 * rgn + 432194612 + seed) % 43214241 * (79143569 + seed);
	
	return 1e-10 * (rgn % 1e10);
	
}