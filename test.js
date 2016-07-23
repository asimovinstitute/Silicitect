// move away from sentence limited model
// remove start and end tokens
// letter size is input chunk size?
// vocab equals index to letter?
// kill all function factories: rowPluck, matrix multiply in graph
// move matrix operations to matrix class
// add different cost functions than shannon entropy
// restructure classes
// optimise? extra matrix random method
// split the whole backward thingy
// add text priming

var temperature = 1.0;
var reguliser = 1e-6;
var learningRate = 0.01;
var clipValue = 5.0;
var generator = "lstm";
var hiddenSizes = [20, 20];
var letterSize = 5;

var epochSize = 0;
var inputSize = 0;
var outputSize = 0;
var letterToIndex = {};
var indexToLetter = {};
var vocabulary = [];
var sentences = [];
var model = {};
var solver = null;

function init (e) {
	
	var totalText = e.responseText.split("\n");
	
	for (var b = 0; b < totalText.length; b++) {
		
		var txt = totalText[b];
		
		for (var a = 0; a < txt.length; a++) {
			
			var char = txt.charAt(a);
			
			if (letterToIndex[char]) continue;
			
			vocabulary.push(char);
			
			letterToIndex[char] = vocabulary.length;
			indexToLetter[vocabulary.length] = char;
			
		}
		
		sentences.push(txt);
		
	}
	
	epochSize = sentences.length;
	inputSize = vocabulary.length + 1;
	outputSize = vocabulary.length + 1;
	
	solver = new Solver();
	model = initModel();
	
	for (var b = 0; b < 2; b++) {
		
		generator = !b ? "lstm" : "rnn";
		
		solver = new Solver();
		model = initModel();
		
		for (var a = 0; a < 100; a++) {
			
			pass();
			
			// Art.doWrite(0, predictSentence(model, temperature, 50) + "\n");
			
			if (a % 5 == 0) console.log(predictSentence(model, temperature, 50));
			
		}
		
	}
	
}

function predictSentence (model, temperature) {
	
	var graph = new Graph(false);
	var sentence = "";
	var log = 0;
	var previous = {};
	var forward = {};
	
	for (var a = 0; a < 200; a++) {
		
		var input = sentence.length == 0 ? 0 : letterToIndex[sentence.charAt(sentence.length - 1)];
		
		forward = forwardIndex(graph, model, input, previous);
		previous = forward;
		
		for (var b = 0; b < forward.o.w.length; b++) {
			
			forward.o.w[b] /= temperature;
			
		}
		
		var probabilities = softmax(forward.o);
		
		var index = sampler(probabilities.w);
		
		if (index == 0) break;
		
		sentence += indexToLetter[index];
		
	}
	
	return sentence;
	
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

function pass () {
	
	var sentence = sentences[Math.floor(Math.random() * sentences.length)];
	var cost = computeCost(model, sentence);
	
	cost.graph.backward();
	
	solver.step(model, learningRate, reguliser, clipValue);
	
}

function computeCost (model, sentence) {
	
	var graph = new Graph(true);
	var log = 0;
	var cost = 0;
	var previous = {};
	var forward = {};
	
	for (var a = -1; a < sentence.length; a++) {
		
		var letter = a == -1 ? 0 : letterToIndex[sentence.charAt(a)];
		var nextLetter = a == sentence.length - 1 ? 0 : letterToIndex[sentence.charAt(a + 1)];
		
		forward = forwardIndex(graph, model, letter, previous);
		previous = forward;
		
		var probabilities = softmax(forward.o);
		
		log -= Math.log2(probabilities.w[nextLetter]);
		cost -= Math.log(probabilities.w[nextLetter]);
		
		forward.o.dw = probabilities.w;
		forward.o.dw[nextLetter] -= 1;
		
	}
	
	return {"graph":graph, "ppl":Math.pow(2, log / (sentence.length - 1)), "cost":cost};
	
}

function forwardIndex (graph, model, index, previous) {
	
	var observation = graph.rowPluck(model["Wil"], index);
	
	if (generator == "lstm") return forwardLSTM(graph, model, hiddenSizes, observation, previous);
	if (generator == "rnn") return forwardRNN(graph, model, hiddenSizes, observation, previous);
	
}

function forwardRNN (graph, model, hiddenSizes, observation, previous) {
	
	var hiddenPrevious = [];
	
	if (previous.h) {
		
		hiddenPrevious = previous.h;
		
	} else {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			hiddenPrevious.push(new Matrix(hiddenSizes[a], 1, 0, 0));
			
		}
		
	}
	
	var hidden = [];
	
	for (var a = 0; a < hiddenSizes.length; a++) {
		
		var input = a == 0 ? observation : hidden[a - 1];
		
		var h0 = graph.multiply(model["Wxh" + a], input);
		var h1 = graph.multiply(model["Whh" + a], hiddenPrevious[a]);
		var hiddenValue = graph.rectifier(graph.add(graph.add(h0, h1), model["bhh" + a]));
		
		hidden.push(hiddenValue);
		
	}
	
	var output = graph.add(graph.multiply(model["Whd"], hidden[hidden.length - 1]), model["bd"]);
	
	return {"h":hidden, "o":output};
	
}

function forwardLSTM (graph, model, hiddenSizes, observation, previous) {
	
	var hiddenPrevious = [];
	var cellPrevious = [];
	
	if (previous.h) {
		
		hiddenPrevious = previous.h;
		cellPrevious = previous.c;
		
	} else {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			hiddenPrevious.push(new Matrix(hiddenSizes[a], 1, 0, 0));
			cellPrevious.push(new Matrix(hiddenSizes[a], 1, 0, 0));
			
		}
		
	}
	
	var hidden = [];
	var cell = [];
	
	for (var a = 0; a < hiddenSizes.length; a++) {
		
		var input = a == 0 ? observation : hidden[a - 1];
		
		var h0 = graph.multiply(model["Wix" + a], input);
		var h1 = graph.multiply(model["Wih" + a], hiddenPrevious[a]);
		var inputGate = graph.sigmoid(graph.add(graph.add(h0, h1), model["bi" + a]));
		
		var h2 = graph.multiply(model["Wfx" + a], input);
		var h3 = graph.multiply(model["Wfh" + a], hiddenPrevious[a]);
		var forgetGate = graph.sigmoid(graph.add(graph.add(h2, h3), model["bf" + a]));
		
		var h4 = graph.multiply(model["Wox" + a], input);
		var h5 = graph.multiply(model["Woh" + a], hiddenPrevious[a]);
		var outputGate = graph.sigmoid(graph.add(graph.add(h4, h5), model["bo" + a]));
		
		var h6 = graph.multiply(model["Wcx" + a], input);
		var h7 = graph.multiply(model["Wch" + a], hiddenPrevious[a]);
		var cellWrite = graph.hyperbolicTangent(graph.add(graph.add(h6, h7), model["bc" + a]));
		
		var retain = graph.feedlessMultiply(forgetGate, cellPrevious[a]);
		var write = graph.feedlessMultiply(inputGate, cellWrite);
		
		var cellValue = graph.add(retain, write);
		var hiddenValue = graph.feedlessMultiply(outputGate, graph.hyperbolicTangent(cellValue));
		
		hidden.push(hiddenValue);
		cell.push(cellValue);
		
	}
	
	var output = graph.add(graph.multiply(model["Whd"], hidden[hidden.length - 1]), model["bd"]);
	
	return {"h":hidden, "c":cell, "o":output};
	
}

function initModel () {
	
	var model = {"Wil":new Matrix(inputSize, letterSize, 0, 0.08)};
	
	if (generator == "rnn") {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			var prevSize = a == 0 ? letterSize : hiddenSizes[a - 1];
			
			model["Wxh" + a] = new Matrix(hiddenSizes[a], prevSize, 0, 0.08);
			model["Whh" + a] = new Matrix(hiddenSizes[a], hiddenSizes[a], 0, 0.08);
			model["bhh" + a] = new Matrix(hiddenSizes[a], 1, 0, 0);
			
		}
		
		model["Whd"] = new Matrix(outputSize, hiddenSizes[hiddenSizes.length - 1], 0, 0.08);
		model["bd"] = new Matrix(outputSize, 1, 0, 0);
		
	} else if (generator == "lstm") {
		
		for (var a = 0; a < hiddenSizes.length; a++) {
			
			var prevSize = a == 0 ? letterSize : hiddenSizes[a - 1];
			
			model['Wix' + a] = new Matrix(hiddenSizes[a], prevSize, 0, 0.08);
			model['Wih' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a], 0, 0.08);
			model['bi' + a] = new Matrix(hiddenSizes[a], 1, 0, 0);
			
			model['Wfx' + a] = new Matrix(hiddenSizes[a], prevSize, 0, 0.08);
			model['Wfh' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a], 0, 0.08);
			model['bf' + a] = new Matrix(hiddenSizes[a], 1, 0, 0);
			
			model['Wox' + a] = new Matrix(hiddenSizes[a], prevSize, 0, 0.08);
			model['Woh' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a], 0, 0.08);
			model['bo' + a] = new Matrix(hiddenSizes[a], 1, 0, 0);
			
			model['Wcx' + a] = new Matrix(hiddenSizes[a], prevSize, 0, 0.08);
			model['Wch' + a] = new Matrix(hiddenSizes[a], hiddenSizes[a], 0, 0.08);
			model['bc' + a] = new Matrix(hiddenSizes[a], 1, 0, 0);
			
		}
		
		model["Whd"] = new Matrix(outputSize, hiddenSizes[hiddenSizes.length - 1], 0, 0.08);
		model["bd"] = new Matrix(outputSize, 1, 0, 0);
		
	}
	
	return model;
	
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

Stecy.setup = function () {
	
	Art.title = "test";
	
};

Art.ready = function () {
	
	Stecy.loadFile("input.txt", init);
	
	Art.doStyle(0, "whiteSpace", "pre");
	
};

(function () {
	
	Solver = function () {
		
		this.decay = 0.999;
		this.smoothing = 1e-8;
		this.lastWeights = {};
		
	};
	
	Solver.prototype.step = function (model, learningRate, reguliser, clipValue) {
		
		for (var a in model) {
			
			if (!this.lastWeights[a]) this.lastWeights[a] = new Matrix(model[a].n, model[a].d, 0, 0);
			
			var ma = model[a];
			var mb = this.lastWeights[a];
			
			for (var b = 0; b < ma.w.length; b++) {
				
				mb.w[b] = mb.w[b] * this.decay + (1 - this.decay) * ma.dw[b] * ma.dw[b];
				
				var clippedValue = Math.max(-clipValue, Math.min(clipValue, ma.dw[b]));
				
				ma.w[b] += -learningRate * clippedValue / Math.sqrt(mb.w[b] + this.smoothing) - reguliser * ma.w[b];
				ma.dw[b] = 0;
				
			}
			
		}
		
	};
	
	Matrix = function (n, d, base, range) {
		
		this.n = n;
		this.d = d;
		this.w = [];
		this.dw = [];
		
		for (var a = 0; a < n * d; a++) {
			
			this.w[a] = base + range * Math.random();
			this.dw[a] = 0;
			
		}
		
	};
	
	Graph = function (needsBackprop) {
		
		this.needsBackprop = needsBackprop;
		this.backprop = [];
		
	};
	
	Graph.prototype.backward = function () {
		
		for (var a = this.backprop.length - 1; a > -1; a--) {
			
			this.backprop[a]();
			
		}
		
	};
	
	Graph.prototype.multiply = function (ma, mb) {
		
		if (ma.d != mb.n) throw new Error("wrong dimensions");
		
		var out = new Matrix(ma.n, mb.d, 0, 0);
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				out.w[mb.d * a + b] = 0;
				
				for (var c = 0; c < ma.d; c++) {
					
					out.w[mb.d * a + b] += ma.w[ma.d * a + c] * mb.w[mb.d * c + b];
					
				}
				
			}
			
		}
		
		if (this.needsBackprop) {
			
			var backward = function () {
				
				for (var a = 0; a < ma.n; a++) {
					
					for (var b = 0; b < mb.d; b++) {
						
						for (var c = 0; c < ma.d; c++) {
							
							ma.dw[ma.d * a + c] += mb.w[mb.d * c + b] * out.dw[mb.d * a + b];
							mb.dw[mb.d * c + b] += ma.w[ma.d * a + c] * out.dw[mb.d * a + b];
							
						}
						
					}
					
				}
				
			};
			
			this.backprop.push(backward);
			
		}
		
		return out;
		
	};
	
	Graph.prototype.feedlessMultiply = function (ma, mb) {
		
		var out = new Matrix(ma.n, ma.d, 0, 0);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = ma.w[a] * mb.w[a];
			
		}
		
		if (this.needsBackprop) {
			
			var backward = function () {
				
				for (var a = 0; a < ma.w.length; a++) {
					
					ma.dw[a] += mb.w[a] * out.dw[a];
					mb.dw[a] += ma.w[a] * out.dw[a];
					
				}
				
			};
			
			this.backprop.push(backward);
			
		}
		
		return out;
		
	};
	
	Graph.prototype.add = function (ma, mb) {
		
		var out = new Matrix(ma.n, ma.d, 0, 0);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = ma.w[a] + mb.w[a];
			
		}
		
		if (this.needsBackprop) {
			
			var backward = function () {
				
				for (var a = 0; a < ma.w.length; a++) {
					
					ma.dw[a] += out.dw[a];
					mb.dw[a] += out.dw[a];
					
				}
				
			};
			
			this.backprop.push(backward);
			
		}
		
		return out;
		
	};
	
	Graph.prototype.sigmoid = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = 1 / (1 + Math.exp(-ma.w[a]));
			
		}
		
		if (this.needsBackprop) {
			
			var backward = function () {
				
				for (var a = 0; a < ma.w.length; a++) {
					
					ma.dw[a] += out.w[a] * (1 - out.w[a]) * out.dw[a];
					
				}
				
			};
			
			this.backprop.push(backward);
			
		}
		
		return out;
		
	};
	
	Graph.prototype.rectifier = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = Math.max(0, ma.w[a]);
			
		}
		
		if (this.needsBackprop) {
			
			var backward = function () {
				
				for (var a = 0; a < ma.w.length; a++) {
					
					ma.dw[a] += ma.w[a] > 0 ? out.dw[a] : 0;
					
				}
				
			};
			
			this.backprop.push(backward);
			
		}
		
		return out;
		
	};
	
	Graph.prototype.hyperbolicTangent = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = Math.tanh(ma.w[a]);
			
		}
		
		if (this.needsBackprop) {
			
			var backward = function () {
				
				for (var a = 0; a < ma.w.length; a++) {
					
					ma.dw[a] += (1 - out.w[a] * out.w[a]) * out.dw[a];
					
				}
				
			};
			
			this.backprop.push(backward);
			
		}
		
		return out;
		
	};
	
	Graph.prototype.rowPluck = function (ma, row) {
		
		var out = new Matrix(ma.d, 1, 0, 0);
		
		for (var a = 0; a < ma.d; a++) {
			
			out.w[a] = ma.w[ma.d * row + a];
			
		}
		
		if (this.needsBackprop) {
			
			var backward = function () {
				
				for (var a = 0; a < ma.d; a++) {
					
					ma.dw[ma.d * row + a] += out.dw[a];
					
				}
				
			};
			
			this.backprop.push(backward);
			
		}
		
		return out;
		
	};
	
})();