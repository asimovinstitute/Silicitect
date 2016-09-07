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

(function () {
	
	Silicitect = function (initialiser, updater) {
		
		this.reguliser = 1e-8;
		this.learningRate = 0.001;
		this.clipValue = 5;
		this.decay = 0.95;
		this.decayLinear = 0.98;
		this.epsilon = 1e-8;
		this.optimiser = Silicitect.rmspropOptimiser;
		
		this.batchTime = 0;
		this.totalLoss = 0;
		this.backprop = [];
		this.network = {};
		this.recordBackprop = false;
		this.initialise = initialiser;
		this.update = updater;
		
		this.initialise();
		this.networkMemory = Matrix.c;
		
	};
	
	Silicitect.logLoss = 0;
	Silicitect.linearLoss = 1;
	Silicitect.binaryLoss = 2;
	
	Silicitect.rmspropOptimiser = 0;
	Silicitect.adamOptimiser = 1;
	
	Silicitect.prototype.startLearningSession = function () {
		
		this.totalLoss = 0;
		this.recordBackprop = true;
		this.batchTime = new Date();
		
	};
	
	Silicitect.prototype.endLearningSession = function () {
		
		this.recordBackprop = false;
		this.batchTime = new Date() - this.batchTime;
		
	};
	
	Silicitect.prototype.backpropagate = function () {
		
		for (var a = this.backprop.length - 1; a > -1; a -= 2) {
			
			if (this.backprop[a].length == 1) this.backprop[a - 1](this.backprop[a][0]);
			if (this.backprop[a].length == 2) this.backprop[a - 1](this.backprop[a][0], this.backprop[a][1]);
			if (this.backprop[a].length == 3) this.backprop[a - 1](this.backprop[a][0], this.backprop[a][1], this.backprop[a][2]);
			
		}
		
		var invDecay = 1 - this.decay;
		var invDecayLinear = 1 - this.decayLinear;
		
		if (this.optimiser == Silicitect.rmspropOptimiser) {
			
			for (var a in this.network) {
				
				var ma = this.network[a];
				
				for (var b = 0; b < ma.l; b++) {
					
					Matrix.vw[ma.i + b] = Matrix.vw[ma.i + b] * this.decay + invDecay * Matrix.dw[ma.i + b] * Matrix.dw[ma.i + b];
					
					var clippedValue = Math.max(-this.clipValue, Math.min(this.clipValue, Matrix.dw[ma.i + b]));
					
					Matrix.w[ma.i + b] -= (this.learningRate * clippedValue) / Math.sqrt(Matrix.vw[ma.i + b] + this.epsilon) + this.reguliser * Matrix.dw[ma.i + b];
					Matrix.dw[ma.i + b] = 0;
					
				}
				
			}
			
		} else if (this.optimiser == Silicitect.adamOptimiser) {
			
			for (var a in this.network) {
				
				var ma = this.network[a];
				
				for (var b = 0; b < ma.l; b++) {
					
					var clippedValue = Math.max(-this.clipValue, Math.min(this.clipValue, Matrix.dw[ma.i + b]));
					
					Matrix.mw[ma.i + b] = Matrix.mw[ma.i + b] * this.decayLinear + invDecayLinear * clippedValue;
					Matrix.vw[ma.i + b] = Matrix.vw[ma.i + b] * this.decay + invDecay * Matrix.dw[ma.i + b] * Matrix.dw[ma.i + b];
					
					Matrix.w[ma.i + b] -= (this.learningRate * (Matrix.mw[ma.i + b] / invDecayLinear)) /
											(Math.sqrt((Matrix.vw[ma.i + b] / invDecay) + this.epsilon) + this.epsilon) +
											this.reguliser * Matrix.dw[ma.i + b];
					Matrix.dw[ma.i + b] = 0;
					
				}
				
			}
			
		}
		
		this.backprop = [];
		this.flush();
		
	};
	
	Silicitect.prototype.flush = function () {
		
		for (var a = Matrix.c; a > this.networkMemory - 1; a--) {
			
			Matrix.w[a] = 0;
			Matrix.dw[a] = 0;
			
		}
		
		Matrix.c = this.networkMemory;
		
	};
	
	Silicitect.prototype.computeLoss = function (lossTarget, desiredValues, squashFunction, lossFunction) {
		
		var squashed = squashFunction(this.network[lossTarget]);
		var sum = 0;
		
		for (var a = 0; a < squashed.l; a++) {
			
			Matrix.dw[this.network[lossTarget].i + a] = -1 * (Matrix.w[this.network[desiredValues].i + a] - Matrix.w[squashed.i + a]);
			
		}
		
		if (lossFunction == Silicitect.logLoss) {
			
			for (var a = 0; a < squashed.l; a++) {
				
				sum += -Math.log(1e-10 + Math.abs(1 - Matrix.w[this.network[desiredValues].i + a] - Matrix.w[squashed.i + a]));
				
			}
			
		} else if (lossFunction == Silicitect.linearLoss) {
			
			for (var a = 0; a < squashed.l; a++) {
				
				sum += Math.abs(Matrix.w[this.network[desiredValues].i + a] - Matrix.w[squashed.i + a]);
				
			}
			
		} else if (lossFunction == Silicitect.binaryLoss) {
			
			for (var a = 0; a < squashed.l; a++) {
				
				sum += Math.round(Math.abs(Matrix.w[this.network[desiredValues].i + a] - Matrix.w[squashed.i + a]));
				
			}
			
		}
		
		this.totalLoss += sum;
		
		return sum;
		
	};
	
	Random = {rgn:0, seed:0};
	
	Random.uniform = function () {
		
		Random.rgn = (4321421413 * Random.rgn + 432194612 + Random.seed) % 43214241 * (79143569 + Random.seed);
		
		return 1e-10 * (Random.rgn % 1e10);
		
	};
	
	TextParser = function (text, characterSet) {
		
		this.text = text;
		this.charToIndex = {};
		this.chars = characterSet;
		
		for (var a = 0; a < this.chars.length; a++) {
			
			this.charToIndex[this.chars.charAt(a)] = a;
			
		}
		
		for (var a = 0; a < this.text.length; a++) {
			
			var char = this.text.charAt(a);
			
			if (1 + this.charToIndex[char]) continue;
			
			this.charToIndex[char] = this.chars.length;
			this.chars += char;
			
		}
		
	};
	
	TextParser.predefinedCharacterSet = "!@#$%^&*()_+{}\":|?><~±§¡€£¢∞œŒ∑´®†¥øØπ∏¬˚∆åÅßΩéúíóáÉÚÍÓÁëüïöäËÜÏÖÄ™‹›ﬁﬂ‡°·—≈çÇ√-=[];',.\\/`µ≤≥„‰◊ˆ˜¯˘¿⁄\n\t";
	TextParser.predefinedCharacterSet += "1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	
	Matrix = function (n, d) {
		
		this.n = n;
		this.d = d;
		this.l = n * d;
		this.i = Matrix.c;
		
		Matrix.c += this.l;
		
	};
	
	Matrix.c = 0;
	Matrix.w = new Float64Array(1e7);
	Matrix.dw = new Float64Array(1e7);
	Matrix.vw = new Float64Array(1e7);
	Matrix.mw = new Float64Array(1e7);
	
	
	Matrix.prototype.randomiseUniform = function () {
		
		for (var a = 0; a < this.l; a++) Matrix.w[this.i + a] = Random.uniform();
		
		return this;
		
	};
	
	Matrix.prototype.randomiseNormalised = function (base, range) {
		
		for (var a = 0; a < this.l; a++) Matrix.w[this.i + a] = Random.uniform() / Math.sqrt(this.d);
		
		return this;
		
	};
	
	Matrix.prototype.fill = function (valueA) {
		
		for (var a = 0; a < this.l; a++) Matrix.w[this.i + a] = valueA;
		
		return this;
		
	};
	
	Matrix.prototype.fillExcept = function (valueA, index, valueB) {
		
		for (var a = 0; a < this.l; a++) Matrix.w[this.i + a] = a == index ? valueB : valueA;
		
		return this;
		
	};
	
	Matrix.silicitect = null;
	
	Matrix.scalar = function (ma, scale) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[ma.i + a] *= scale;
			
		}
		
		return out;
		
	};
	
	Matrix.softmax = function (ma, temp) {
		
		var out = new Matrix(ma.n, ma.d);
		var max = -1e10;
		var sum = 0;
		
		if (temp) {
			
			for (var a = 0; a < ma.l; a++) Matrix.w[ma.i + a] /= temp;
			
		}
		
		for (var a = 0; a < ma.l; a++) {
			
			if (Matrix.w[ma.i + a] > max) max = Matrix.w[ma.i + a];
			
		}
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = Math.exp(Matrix.w[ma.i + a] - max);
			
			sum += Matrix.w[out.i + a];
			
		}
		
		for (var a = 0; a < ma.l; a++) Matrix.w[out.i + a] /= sum;
		
		return out;
		
	};
	
	Matrix.sampleMax = function (ma) {
		
		var highest = 0;
		
		for (var a = 1; a < ma.l; a++) {
			
			if (Matrix.w[ma.i + a] > ma.w[highest]) highest = a;
			
		}
		
		return highest;
		
	};
	
	Matrix.sampleRandomSum = function (ma) {
		
		var random = Math.random();
		var sum = 0;
		
		for (var a = 0; a < ma.l; a++) {
			
			sum += Matrix.w[ma.i + a];
			
			if (sum > random) return a;
			
		}
		
		return a - 1;
		
	};
	
	Matrix.invert = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = 1 - Matrix.w[ma.i + a];
			
		}
		
		return out;
		
	};
	
	Matrix.nothing = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = Matrix.w[ma.i + a];
			
		}
		
		return out;
		
	};
	
	Matrix.multiply = function (ma, mb) {
		
		var out = new Matrix(ma.n, mb.d);
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				Matrix.w[out.i + mb.d * a + b] = 0;
				
				for (var c = 0; c < ma.d; c++) {
					
					Matrix.w[out.i + mb.d * a + b] += Matrix.w[ma.i + ma.d * a + c] * Matrix.w[mb.i + mb.d * c + b];
					
				}
				
			}
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.multiplyBackward, [ma, mb, out]);
		
		return out;
		
	};
	
	Matrix.multiplyBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				for (var c = 0; c < ma.d; c++) {
					
					Matrix.dw[ma.i + ma.d * a + c] += Matrix.w[mb.i + mb.d * c + b] * Matrix.dw[out.i + mb.d * a + b];
					Matrix.dw[mb.i + mb.d * c + b] += Matrix.w[ma.i + ma.d * a + c] * Matrix.dw[out.i + mb.d * a + b];
					
				}
				
			}
			
		}
		
	};
	
	Matrix.elementMultiply = function (ma, mb) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = Matrix.w[ma.i + a] * Matrix.w[mb.i + a];
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.elementMultiplyBackward, [ma, mb, out]);
		
		return out;
		
	};
	
	Matrix.elementMultiplyBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.dw[ma.i + a] += Matrix.w[mb.i + a] * Matrix.dw[out.i + a];
			Matrix.dw[mb.i + a] += Matrix.w[ma.i + a] * Matrix.dw[out.i + a];
			
		}
		
	};
	
	Matrix.add = function (ma, mb) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = Matrix.w[ma.i + a] + Matrix.w[mb.i + a];
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.addBackward, [ma, mb, out]);
		
		return out;
		
	};
	
	Matrix.addBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.dw[ma.i + a] += Matrix.dw[out.i + a];
			Matrix.dw[mb.i + a] += Matrix.dw[out.i + a];
			
		}
		
	};
	
	Matrix.sigmoid = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = 1 / (1 + Math.exp(-Matrix.w[ma.i + a]));
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.sigmoidBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.sigmoidBackward = function (ma, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.dw[ma.i + a] += Matrix.w[out.i + a] * (1 - Matrix.w[out.i + a]) * Matrix.dw[out.i + a];
			
		}
		
	};
	
	Matrix.rectifiedLinear = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = Math.max(0, Matrix.w[ma.i + a]);
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.rectifiedLinearBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.rectifiedLinearBackward = function (ma, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.dw[ma.i + a] += Matrix.w[ma.i + a] > 0 ? Matrix.dw[out.i + a] : 0;
			
		}
		
	};
	
	Matrix.hyperbolicTangent = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.w[out.i + a] = Math.tanh(Matrix.w[ma.i + a]);
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.hyperbolicTangentBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.hyperbolicTangentBackward = function (ma, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			Matrix.dw[ma.i + a] += (1 - Matrix.w[out.i + a] * Matrix.w[out.i + a]) * Matrix.dw[out.i + a];
			
		}
		
	};
	
	Matrix.rowPluck = function (ma, row) {
		
		var out = new Matrix(ma.d, 1);
		
		for (var a = 0; a < ma.d; a++) {
			
			Matrix.w[out.i + a] = Matrix.w[ma.i + ma.d * row + a];
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.rowPluckBackward, [ma, out, row]);
		
		return out;
		
	};
	
	Matrix.rowPluckBackward = function (ma, out, row) {
		
		for (var a = 0; a < ma.d; a++) {
			
			Matrix.dw[ma.i + ma.d * row + a] += Matrix.dw[out.i + a];
			
		}
		
	}
	
})();