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
	
	Silicitect = function (initialiseFunction, updateFunction) {
		
		this.reguliser = 1e-8;
		this.learningRate = 0.1;
		this.clipValue = 5;
		this.decayRate = 0.95;
		
		this.batchTime = 0;
		this.totalLoss = 0;
		this.backprop = [];
		this.recordBackprop = false;
		this.lastWeights = {};
		this.network = {};
		this.initialiseFunction = initialiseFunction;
		this.updateFunction = updateFunction;
		
		this.initialise();
		
	};
	
	Silicitect.prototype.initialise = function () {
		
		this.initialiseFunction(this.network);
		
		for (var a in this.network) {
			
			this.lastWeights[a] = new Matrix(this.network[a].n, this.network[a].d);
			
		}
		
		return this;
		
	};
	
	Silicitect.prototype.startLearningSession = function () {
		
		this.totalLoss = 0;
		this.recordBackprop = true;
		this.batchTime = new Date();
		
	};
	
	Silicitect.prototype.endLearningSession = function () {
		
		this.recordBackprop = false;
		this.batchTime = new Date() - this.batchTime;
		
	};
	
	Silicitect.prototype.update = function () {
		
		this.updateFunction(this.network);
		
		return this;
		
	};
	
	Silicitect.prototype.backpropagate = function () {
		
		for (var a = this.backprop.length - 1; a > -1; a -= 2) {
			
			if (this.backprop[a].length == 1) this.backprop[a - 1](this.backprop[a][0]);
			if (this.backprop[a].length == 2) this.backprop[a - 1](this.backprop[a][0], this.backprop[a][1]);
			if (this.backprop[a].length == 3) this.backprop[a - 1](this.backprop[a][0], this.backprop[a][1], this.backprop[a][2]);
			
		}
		
		for (var a in this.network) {
			
			var ma = this.network[a];
			var mb = this.lastWeights[a];
			
			for (var b = 0; b < ma.w.length; b++) {
				
				mb.w[b] = mb.w[b] * this.decayRate + (1 - this.decayRate) * ma.dw[b] * ma.dw[b];
				
				var clippedValue = Math.max(-this.clipValue, Math.min(this.clipValue, ma.dw[b]));
				
				ma.w[b] += -this.learningRate * clippedValue / Math.sqrt(mb.w[b] + 1e-8) - this.reguliser * ma.w[b];
				ma.dw[b] = 0;
				
			}
			
		}
		
		this.backprop = [];
		
	};
	
	Silicitect.prototype.computeLoss = function (lossTarget, desiredValues, squashFunction, lossFunction) {
		
		var squashed = squashFunction(this.network[lossTarget]);
		var sum = 0;
		
		for (var a = 0; a < squashed.w.length; a++) {
			//?
			this.network[lossTarget].dw[a] = -1 * (this.network[desiredValues].w[a] - squashed.w[a]);
			
		}
		
		if (lossFunction == Silicitect.logLoss) {
			
			for (var a = 0; a < squashed.w.length; a++) {
				
				sum += -Math.log(Math.abs(1 - this.network[desiredValues].w[a] - squashed.w[a]));
				
			}
			
		} else if (lossFunction == Silicitect.linearLoss) {
			
			for (var a = 0; a < squashed.w.length; a++) {
				
				sum += Math.abs(this.network[desiredValues].w[a] - squashed.w[a]);
				
			}
			
		} else if (lossFunction == Silicitect.binaryLoss) {
			
			for (var a = 0; a < squashed.w.length; a++) {
				
				sum += Math.round(Math.abs(this.network[desiredValues].w[a] - squashed.w[a]));
				
			}
			
		}
		
		this.totalLoss += sum;
		
		return sum;
		
	};
	
	Silicitect.logLoss = 0;
	Silicitect.linearLoss = 1;
	Silicitect.binaryLoss = 2;
	
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
	
	TextParser.predefinedCharacterSet = "!@#$%^&*()_+{}\":|?><~±§¡€£¢∞œŒ∑´®†¥øØπ∏¬˚∆åÅßΩéúíóáÉÚÍÓÁëüïöäËÜÏÖÄ™‹›ﬁﬂ‡°·—≈çÇ√-=[];',.\\/`µ≤≥„‰◊ˆ˜¯˘¿⁄\n\t" + 
										"1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	
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
		
		for (var a = 0; a < this.w.length; a++) this.w[a] = uniform();
		
		return this;
		
	};
	
	Matrix.prototype.randomiseNormalised = function (base, range) {
		
		for (var a = 0; a < this.w.length; a++) this.w[a] = uniform() / Math.sqrt(this.d);
		
		return this;
		
	};
	
	Matrix.prototype.fill = function (va) {
		
		for (var a = 0; a < this.w.length; a++) this.w[a] = va;
		
		return this;
		
	};
	
	Matrix.prototype.fillExcept = function (va, i, vb) {
		
		for (var a = 0; a < this.w.length; a++) this.w[a] = a == i ? vb : va;
		
		return this;
		
	};
	
	Matrix.silicitect = null;
	
	Matrix.scalar = function (ma, scale) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.w[a] *= scale;
			
		}
		
		return out;
		
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
		
		for (var a = 0; a < ma.w.length; a++) out.w[a] /= sum;
		
		return out;
		
	};
	
	Matrix.sampleMax = function (ma) {
		
		var highest = 0;
		
		for (var a = 1; a < ma.w.length; a++) {
			
			if (ma.w[a] > ma.w[highest]) highest = a;
			
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
	
	Matrix.invert = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = 1 - ma.w[a];
			
		}
		
		return out;
		
	};
	
	Matrix.nothing = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = ma.w[a];
			
		}
		
		return out;
		
	};
	
	Matrix.multiply = function (ma, mb) {
		
		var out = new Matrix(ma.n, mb.d);
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				out.w[mb.d * a + b] = 0;
				
				for (var c = 0; c < ma.d; c++) {
					
					out.w[mb.d * a + b] += ma.w[ma.d * a + c] * mb.w[mb.d * c + b];
					
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
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.elementMultiplyBackward, [ma, mb, out]);
		
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
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.addBackward, [ma, mb, out]);
		
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
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.sigmoidBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.sigmoidBackward = function (ma, out) {
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.dw[a] += out.w[a] * (1 - out.w[a]) * out.dw[a];
			
		}
		
	};
	
	Matrix.rectifiedLinear = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = Math.max(0, ma.w[a]);
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.rectifiedLinearBackward, [ma, out]);
		
		return out;
		
	};
	
	Matrix.rectifiedLinearBackward = function (ma, out) {
		
		for (var a = 0; a < ma.w.length; a++) {
			
			ma.dw[a] += ma.w[a] > 0 ? out.dw[a] : 0;
			
		}
		
	};
	
	Matrix.hyperbolicTangent = function (ma) {
		
		var out = new Matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.w.length; a++) {
			
			out.w[a] = Math.tanh(ma.w[a]);
			
		}
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.hyperbolicTangentBackward, [ma, out]);
		
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
		
		if (Matrix.silicitect.recordBackprop) Matrix.silicitect.backprop.push(Matrix.rowPluckBackward, [ma, out, row]);
		
		return out;
		
	};
	
	Matrix.rowPluckBackward = function (ma, out, row) {
		
		for (var a = 0; a < ma.d; a++) {
			
			ma.dw[ma.d * row + a] += out.dw[a];
			
		}
		
	}
	
})();