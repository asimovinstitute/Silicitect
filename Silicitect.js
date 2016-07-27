// float arrays?
// reuse memory
// dynamic interrupts
// save/loading of models
// move matrix operations to matrix class
// add different cost functions than shannon entropy
// restructure classes
// optimise? extra matrix random method
// split the whole backward thingy

var temperature = 1.0;
var reguliser = 0.000001;
var learningRate = 0.01;
var clipValue = 5.0;
var hiddenSizes = [10];
var letterEmbedSize = 3;
var decayRate = 0.97;

var text = "";
var inputSize = 0;
var outputSize = 0;
var letterToIndex = {};
var indexToLetter = [];
var model = {};
var lastWeights = {};
var recordBackprop = false;
var backprop = [];
var characterSet = "analyse";

var characters = "!@#$%^&*()_+{}\":|?><~±§¡€£¢∞œŒ∑´®†¥øØπ∏¬˚∆åÅßΩéúíóáÉÚÍÓÁëüïöäËÜÏÖÄ⁄™‹›ﬁﬂ‡°·—±≈çÇ√-=[];',.\\/`~µ≤≥„‰◊ˆ˜¯˘¿—⁄\n\t" + 
			"1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

function init (e) {
	
	text = e.responseText;
	
	if (characterSet == "predefined") {
		
		for (var a = 0; a < characters.length; a++) {
			
			if (letterToIndex[characters.charAt(a)]) continue;
			
			letterToIndex[characters.charAt(a)] = indexToLetter.length;
			indexToLetter[indexToLetter.length] = characters.charAt(a);
			
		}
		
		for (var a = 0; a < text.length; a++) {
			
			if (!letterToIndex[text.charAt(a)]) {
				
				console.log("Wrong character found, " + text.charAt(a) + " not in " + characterSet);
				return;
				
			}
			
		}
		
	} else if (characterSet == "analyse") {
		
		for (var a = 0; a < text.length; a++) {
			
			var char = text.charAt(a);
			
			if (letterToIndex[char]) continue;
			
			letterToIndex[char] = indexToLetter.length;
			indexToLetter[indexToLetter.length] = char;
			
		}
		
	} else {
		
		console.log("Wrong character set specified");
		return;
		
	}
	
	inputSize = indexToLetter.length;
	outputSize = indexToLetter.length;
	
	initModel("lstm");
	
	batch(1000, 20, 100);
	
}

function updateWeights () {
	
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

function batch (iterations, batchSize, sampleInterval) {
	
	var startTime = new Date();
	var averageTime = 0;
	var averageLoss = 0;
	var sentence = "";
	
	for (var a = 0; a < iterations; a++) {
		
		sentence = text.substr(Math.floor(Math.random() * (text.length - batchSize)), batchSize);
		
		averageLoss += train(sentence);
		
		if (a % sampleInterval == sampleInterval - 1) {
			
			averageTime += new Date() - startTime;
			
			// Art.doWrite(0, a + 1 + "\t" + (averageLoss / sampleInterval).toFixed(2) + "\t" + (new Date() - startTime) + "ms\t" + ask(50, "") + "\n");
			console.log(a + 1, (averageLoss / sampleInterval).toFixed(2), (new Date() - startTime) + "ms", ask(50, ""));
			
			averageLoss = 0;
			startTime = new Date();
			
		}
		
	}
	
	// Art.doWrite(0, "Done training, average: " + Math.round(averageTime / (a / sampleInterval)) + "ms");
	console.log("Done training, average: " + Math.round(averageTime / (a / sampleInterval)) + "ms");
	
}

function ask (length, prime) {
	
	recordBackprop = false;
	
	var sentence = prime;
	var log = 0;
	var previous = {};
	var forward = {};
	
	for (var a = 0; a < prime.length; a++) {
		
		var letter = letterToIndex[prime.charAt(a)];
		
		forward = forwardLSTM(letter, previous);
		previous = forward;
		
	}
	
	for (var a = 0; a < length; a++) {
		
		var inputLetter = sentence.length == 0 ? 0 : letterToIndex[sentence.charAt(sentence.length - 1)];
		
		forward = forwardLSTM(inputLetter, previous);
		previous = forward;
		
		for (var b = 0; b < forward.o.w.length; b++) {
			
			forward.o.w[b] /= temperature;
			
		}
		
		var probabilities = softmax(forward.o);
		var index = sampler(probabilities.w);
		
		sentence += indexToLetter[index];
		
	}
	
	return sentence.slice(prime.length);
	
}

function train (sentence) {
	
	recordBackprop = true;
	backprop = [];
	
	var loss = 0;
	var previous = {};
	var forward = {};
	
	for (var a = 0; a < sentence.length - 1; a++) {
		
		var letter = letterToIndex[sentence.charAt(a)];
		var nextLetter = letterToIndex[sentence.charAt(a + 1)];
		
		if (!(letter + 1)) {
			
			console.log("Found unkown character: " + sentence.charAt(a));
			break;
			
		}
		
		forward = forwardLSTM(letter, previous);
		previous = forward;
		
		var probabilities = softmax(forward.o);
		
		loss -= Math.log(probabilities.w[nextLetter]);
		
		forward.o.dw = probabilities.w;
		forward.o.dw[nextLetter] -= 1;
		
	}
	
	backward();
	
	updateWeights();
	
	return loss;
	
}

function forwardRNN (letter, previous) {
	
	var observation = rowPluck(model["Wil"], letter);
	var hiddenPrevious = [];
	
	if (previous.h) {
		
		hiddenPrevious = previous.h;
		
	} else {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			hiddenPrevious.push(new Matrix(hiddenSizes[a], 1));
			
		}
		
	}
	
	var hidden = [];
	
	for (var a = 0; a < hiddenSizes.length; a++) {
		
		var input = a == 0 ? observation : hidden[a - 1];
		
		var h0 = multiply(model["Wxh" + a], input);
		var h1 = multiply(model["Whh" + a], hiddenPrevious[a]);
		var hiddenValue = rectifier(add(add(h0, h1), model["bhh" + a]));
		
		hidden.push(hiddenValue);
		
	}
	
	var output = add(multiply(model["Whd"], hidden[hidden.length - 1]), model["bd"]);
	
	return {"h":hidden, "o":output};
	
}

function forwardLSTM (letter, previous) {
	
	var observation = rowPluck(model["Wil"], letter);
	var hiddenPrevious = [];
	var cellPrevious = [];
	
	if (previous.h) {
		
		hiddenPrevious = previous.h;
		cellPrevious = previous.c;
		
	} else {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			hiddenPrevious.push(new Matrix(hiddenSizes[a], 1));
			cellPrevious.push(new Matrix(hiddenSizes[a], 1));
			
		}
		
	}
	
	var hidden = [];
	var cell = [];
	
	for (var a = 0; a < hiddenSizes.length; a++) {
		
		var input = a == 0 ? observation : hidden[a - 1];
		
		var h0 = multiply(model["Wix" + a], input);
		var h1 = multiply(model["Wih" + a], hiddenPrevious[a]);
		var inputGate = sigmoid(add(add(h0, h1), model["bi" + a]));
		
		var h2 = multiply(model["Wfx" + a], input);
		var h3 = multiply(model["Wfh" + a], hiddenPrevious[a]);
		var forgetGate = sigmoid(add(add(h2, h3), model["bf" + a]));
		
		var h4 = multiply(model["Wox" + a], input);
		var h5 = multiply(model["Woh" + a], hiddenPrevious[a]);
		var outputGate = sigmoid(add(add(h4, h5), model["bo" + a]));
		
		var h6 = multiply(model["Wcx" + a], input);
		var h7 = multiply(model["Wch" + a], hiddenPrevious[a]);
		var cellWrite = hyperbolicTangent(add(add(h6, h7), model["bc" + a]));
		
		var retain = feedlessMultiply(forgetGate, cellPrevious[a]);
		var write = feedlessMultiply(inputGate, cellWrite);
		
		var cellValue = add(retain, write);
		var hiddenValue = feedlessMultiply(outputGate, hyperbolicTangent(cellValue));
		
		hidden.push(hiddenValue);
		cell.push(cellValue);
		
	}
	
	var output = add(multiply(model["Whd"], hidden[hidden.length - 1]), model["bd"]);
	
	return {"h":hidden, "c":cell, "o":output};
	
}

function initModel (generator) {
	
	model = {"Wil":new Matrix(inputSize, letterEmbedSize).randomise(0, 0.08)};
	
	if (generator == "rnn") {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			var prevSize = a == 0 ? letterEmbedSize : hiddenSizes[a - 1];
			
			model["Wxh" + a] = new Matrix(hiddenSizes[a], prevSize).randomise(0, 0.08);
			model["Whh" + a] = new Matrix(hiddenSizes[a], hiddenSizes[a]).randomise(0, 0.08);
			model["bhh" + a] = new Matrix(hiddenSizes[a], 1);
			
		}
		
		model["Whd"] = new Matrix(outputSize, hiddenSizes[hiddenSizes.length - 1]).randomise(0, 0.08);
		model["bd"] = new Matrix(outputSize, 1);
		
	} else if (generator == "lstm") {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			var prevSize = a == 0 ? letterEmbedSize : hiddenSizes[a - 1];
			
			model['Wix' + a] = new Matrix(hiddenSizes[a], prevSize).randomise(0, 0.08);
			model['Wih' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a]).randomise(0, 0.08);
			model['bi' + a] = new Matrix(hiddenSizes[a], 1);
			
			model['Wfx' + a] = new Matrix(hiddenSizes[a], prevSize).randomise(0, 0.08);
			model['Wfh' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a]).randomise(0, 0.08);
			model['bf' + a] = new Matrix(hiddenSizes[a], 1);
			
			model['Wox' + a] = new Matrix(hiddenSizes[a], prevSize).randomise(0, 0.08);
			model['Woh' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a]).randomise(0, 0.08);
			model['bo' + a] = new Matrix(hiddenSizes[a], 1);
			
			model['Wcx' + a] = new Matrix(hiddenSizes[a], prevSize).randomise(0, 0.08);
			model['Wch' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a]).randomise(0, 0.08);
			model['bc' + a] = new Matrix(hiddenSizes[a], 1);
			
		}
		
		model["Whd"] = new Matrix(outputSize, hiddenSizes[hiddenSizes.length - 1]).randomise(0, 0.08);
		model["bd"] = new Matrix(outputSize, 1);
		
	}
	
}

function softmax (ma) {
	
	var out = new Matrix(ma.n, ma.d);
	var max = -1e10;
	var sum = 0;
	
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
	
}

function sampler (w) {
	
	var random = Math.random();
	var sum = 0;
	
	for (var a = 0; a < w.length; a++) {
		
		sum += w[a];
		
		if (sum > random) return a;
		
	}
	
	return a.length - 1;
	
}

Stecy.setup = function () {
	
	Art.title = "Silicitect";
	
};

Art.ready = function () {
	
	Stecy.loadFile("input/simple.txt", init);
	
	Art.doStyle(0, "whiteSpace", "pre", "font", "20px monospace", "tabSize", "6");
	
};

(function () {
	
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
	
	Matrix.prototype.randomise = function (base, range) {
		
		for (var a = 0; a < this.n * this.d; a++) {
			
			this.w[a] = base + range * Math.random();
			
		}
		
		return this;
		
	};
	
})();

function backward () {
	
	for (var a = backprop.length - 1; a > -1; a -= 2) {
		
		if (backprop[a].length == 1) backprop[a - 1](backprop[a][0]);
		if (backprop[a].length == 2) backprop[a - 1](backprop[a][0], backprop[a][1]);
		if (backprop[a].length == 3) backprop[a - 1](backprop[a][0], backprop[a][1], backprop[a][2]);
		
	}
	
}

function multiply (ma, mb) {
	
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
	
	if (recordBackprop) {
		
		backprop.push(multiplyBackward, [ma, mb, out]);
		
	}
	
	return out;
	
}

function multiplyBackward (ma, mb, out) {
	
	for (var a = 0; a < ma.n; a++) {
		
		for (var b = 0; b < mb.d; b++) {
			
			for (var c = 0; c < ma.d; c++) {
				
				ma.dw[ma.d * a + c] += mb.w[mb.d * c + b] * out.dw[mb.d * a + b];
				mb.dw[mb.d * c + b] += ma.w[ma.d * a + c] * out.dw[mb.d * a + b];
				
			}
			
		}
		
	}
	
}

function feedlessMultiply (ma, mb) {
	
	var out = new Matrix(ma.n, ma.d);
	
	for (var a = 0; a < ma.w.length; a++) {
		
		out.w[a] = ma.w[a] * mb.w[a];
		
	}
	
	if (recordBackprop) {
		
		backprop.push(feedlessMultiplyBackward, [ma, mb, out]);
		
	}
	
	return out;
	
}

function feedlessMultiplyBackward (ma, mb, out) {
	
	for (var a = 0; a < ma.w.length; a++) {
		
		ma.dw[a] += mb.w[a] * out.dw[a];
		mb.dw[a] += ma.w[a] * out.dw[a];
		
	}
	
}

function add (ma, mb) {
	
	var out = new Matrix(ma.n, ma.d);
	
	for (var a = 0; a < ma.w.length; a++) {
		
		out.w[a] = ma.w[a] + mb.w[a];
		
	}
	
	if (recordBackprop) {
		
		backprop.push(addBackward, [ma, mb, out]);
		
	}
	
	return out;
	
}

function addBackward (ma, mb, out) {
	
	for (var a = 0; a < ma.w.length; a++) {
		
		ma.dw[a] += out.dw[a];
		mb.dw[a] += out.dw[a];
		
	}
	
}

function sigmoid (ma) {
	
	var out = new Matrix(ma.n, ma.d);
	
	for (var a = 0; a < ma.w.length; a++) {
		
		out.w[a] = 1 / (1 + Math.exp(-ma.w[a]));
		
	}
	
	if (recordBackprop) {
		
		backprop.push(sigmoidBackward, [ma, out]);
		
	}
	
	return out;
	
}

function sigmoidBackward (ma, out) {
	
	for (var a = 0; a < ma.w.length; a++) {
		
		ma.dw[a] += out.w[a] * (1 - out.w[a]) * out.dw[a];
		
	}
	
}

function rectifier (ma) {
	
	var out = new Matrix(ma.n, ma.d);
	
	for (var a = 0; a < ma.w.length; a++) {
		
		out.w[a] = Math.max(0, ma.w[a]);
		
	}
	
	if (recordBackprop) {
		
		backprop.push(rectifierBackward, [ma, out]);
		
	}
	
	return out;
	
}

function rectifierBackward (ma, out) {
	
	for (var a = 0; a < ma.w.length; a++) {
		
		ma.dw[a] += ma.w[a] > 0 ? out.dw[a] : 0;
		
	}
	
}

function hyperbolicTangent (ma) {
	
	var out = new Matrix(ma.n, ma.d);
	
	for (var a = 0; a < ma.w.length; a++) {
		
		out.w[a] = Math.tanh(ma.w[a]);
		
	}
	
	if (recordBackprop) {
		
		backprop.push(hyperbolicTangentBackward, [ma, out]);
		
	}
	
	return out;
	
}

function hyperbolicTangentBackward (ma, out) {
	
	for (var a = 0; a < ma.w.length; a++) {
		
		ma.dw[a] += (1 - out.w[a] * out.w[a]) * out.dw[a];
		
	}
	
}

function rowPluck (ma, row) {
	
	var out = new Matrix(ma.d, 1);
	
	for (var a = 0; a < ma.d; a++) {
		
		out.w[a] = ma.w[ma.d * row + a];
		
	}
	
	if (recordBackprop) {
		
		backprop.push(rowPluckBackward, [ma, out, row]);
		
	}
	
	return out;
	
}

function rowPluckBackward (ma, out, row) {
	
	for (var a = 0; a < ma.d; a++) {
		
		ma.dw[ma.d * row + a] += out.dw[a];
		
	}
	
}
