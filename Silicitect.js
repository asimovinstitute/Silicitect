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

(function () {
	
	var enumator = 0;
	
	Silicitect = function () {
		
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
		this.net = {};
		this.recordBackprop = false;
		
		this.text = {};
		this.text.raw = "";
		this.text.charToIndex = {};
		this.text.characterSet = "";
		
		this.image = {};
		
		this.sound = {};
		
		this.random = {};
		this.random.lossless = 0;
		this.random.nextGaussian = 0;
		this.random.seed = 0;
		
		this.memCount = 0;
		this.weights = new Float64Array(1e7);
		this.dWeights = new Float64Array(1e7);
		this.vWeights = new Float64Array(1e7);
		this.mWeights = new Float64Array(1e7);
		
		
	};
	
	Silicitect.prototype.init = function (initialiser, memory) {
		
		memory = memory ? memory : 1e7;
		
		this.memCount = 0;
		this.weights = new Float64Array(memory);
		this.dWeights = new Float64Array(memory);
		this.vWeights = new Float64Array(memory);
		this.mWeights = new Float64Array(memory);
		
		initialiser();
		
		this.netMemory = this.memCount;
		
	};
	
	Silicitect.logLoss = enumator++;
	Silicitect.linearLoss = enumator++;
	Silicitect.binaryLoss = enumator++;
	
	Silicitect.rmspropOptimiser = enumator++;
	Silicitect.adamOptimiser = enumator++;
	
	Silicitect.predefinedCharacterSet = enumator++;
	
	Silicitect.matrixMultiplyID = enumator++;
	Silicitect.matrixElementMultiplyID = enumator++;
	Silicitect.matrixAddID = enumator++;
	Silicitect.matrixSigmoidID = enumator++;
	Silicitect.matrixRectifiedLinearID = enumator++;
	Silicitect.matrixHyperbolicTangentID = enumator++;
	
	Silicitect.prototype.startLearningSession = function () {
		
		this.totalLoss = 0;
		this.recordBackprop = true;
		this.batchTime = new Date();
		
	};
	
	Silicitect.prototype.endLearningSession = function () {
		
		this.recordBackprop = false;
		this.batchTime = new Date() - this.batchTime;
		
	};
	
	Silicitect.prototype.viewNet = function (displayWeights) {
		
		var result = "";
		
		for (var a in this.net) {
			
			result += a + ", " + this.net[a].n + " x " + this.net[a].d + "\n";
			
			if (displayWeights) {
				
				for (var b = 0; b < this.net[a].l; b++) {
					
					if (this.weights[this.net[a].i + b] >= 0) result += " ";
					
					result += this.weights[this.net[a].i + b].toFixed(3) + (b % this.net[a].d == this.net[a].d - 1 ? "\n" : "   ");
					
				}
				
			}
			
		}
		
		return result;
		
	};
	
	Silicitect.prototype.backpropagate = function () {
		
		for (var a = this.backprop.length - 1; a > -1; a--) {
			
			if (this.backprop[a] == Silicitect.matrixMultiplyID) {
				
				this.matrixMultiplyBackward(this.backprop[a - 3], this.backprop[a - 2], this.backprop[a - 1]);
				
				a -= 3;
				
			} else if (this.backprop[a] == Silicitect.matrixElementMultiplyID) {
				
				this.matrixElementMultiplyBackward(this.backprop[a - 3], this.backprop[a - 2], this.backprop[a - 1]);
				
				a -= 3;
				
			} else if (this.backprop[a] == Silicitect.matrixAddID) {
				
				this.matrixAddBackward(this.backprop[a - 3], this.backprop[a - 2], this.backprop[a - 1]);
				
				a -= 3;
				
			} else if (this.backprop[a] == Silicitect.matrixSigmoidID) {
				
				this.matrixSigmoidBackward(this.backprop[a - 2], this.backprop[a - 1]);
				
				a -= 2;
				
			} else if (this.backprop[a] == Silicitect.matrixRectifiedLinearID) {
				
				this.matrixRectifiedLinearBackward(this.backprop[a - 2], this.backprop[a - 1]);
				
				a -= 2;
				
			} else if (this.backprop[a] == Silicitect.matrixHyperbolicTangentID) {
				
				this.matrixHyperbolicTangentBackward(this.backprop[a - 2], this.backprop[a - 1]);
				
				a -= 2;
				
			} else {
				console.log(this.backprop[a]);
				return;
			}
			
		}
		
		var invDecay = 1 - this.decay;
		var invDecayLinear = 1 - this.decayLinear;
		
		if (this.optimiser == Silicitect.rmspropOptimiser) {
			
			for (var a in this.net) {
				
				var ma = this.net[a];
				
				for (var b = 0; b < ma.l; b++) {
					
					this.vWeights[ma.i + b] = this.vWeights[ma.i + b] * this.decay + invDecay * this.dWeights[ma.i + b] * this.dWeights[ma.i + b];
					
					var clippedValue = Math.max(-this.clipValue, Math.min(this.clipValue, this.dWeights[ma.i + b]));
					
					this.weights[ma.i + b] -= (this.learningRate * clippedValue) / Math.sqrt(this.vWeights[ma.i + b] + this.epsilon) + this.reguliser * this.dWeights[ma.i + b];
					this.dWeights[ma.i + b] = 0;
					
				}
				
			}
			
		} else if (this.optimiser == Silicitect.adamOptimiser) {
			
			for (var a in this.net) {
				
				var ma = this.net[a];
				
				for (var b = 0; b < ma.l; b++) {
					
					var clippedValue = Math.max(-this.clipValue, Math.min(this.clipValue, this.dWeights[ma.i + b]));
					
					this.mWeights[ma.i + b] = this.mWeights[ma.i + b] * this.decayLinear + invDecayLinear * clippedValue;
					this.vWeights[ma.i + b] = this.vWeights[ma.i + b] * this.decay + invDecay * this.dWeights[ma.i + b] * this.dWeights[ma.i + b];
					
					this.weights[ma.i + b] -= (this.learningRate * (this.mWeights[ma.i + b] / invDecayLinear)) /
											(Math.sqrt((this.vWeights[ma.i + b] / invDecay) + this.epsilon) + this.epsilon) +
											this.reguliser * this.dWeights[ma.i + b];
					this.dWeights[ma.i + b] = 0;
					
				}
				
			}
			
		}
		
		this.backprop = [];
		this.flush();
		
	};
	
	Silicitect.prototype.flush = function () {
		
		for (var a = this.memCount; a > this.netMemory + 1; a--) {
			
			this.weights[a] = 0;
			this.dWeights[a] = 0;
			
		}
		
		this.memCount = this.netMemory + 1;
		
	};
	
	Silicitect.prototype.computeLoss = function (lossTarget, desiredValues, squashFunction, lossFunction) {
		
		var squashed = this[squashFunction](this.net[lossTarget]);
		var sum = 0;
		
		for (var a = 0; a < squashed.l; a++) {
			
			this.dWeights[this.net[lossTarget].i + a] = -1 * (this.weights[this.net[desiredValues].i + a] - this.weights[squashed.i + a]);
			
		}
		
		if (lossFunction == Silicitect.logLoss) {
			
			for (var a = 0; a < squashed.l; a++) {
				
				sum += -Math.log(1e-10 + Math.abs(1 - this.weights[this.net[desiredValues].i + a] - this.weights[squashed.i + a]));
				
			}
			
		} else if (lossFunction == Silicitect.linearLoss) {
			
			for (var a = 0; a < squashed.l; a++) {
				
				sum += Math.abs(this.weights[this.net[desiredValues].i + a] - this.weights[squashed.i + a]);
				
			}
			
		} else if (lossFunction == Silicitect.binaryLoss) {
			
			for (var a = 0; a < squashed.l; a++) {
				
				sum += Math.round(Math.abs(this.weights[this.net[desiredValues].i + a] - this.weights[squashed.i + a]));
				
			}
			
		}
		
		this.totalLoss += sum;
		
		return sum;
		
	};
	
	Silicitect.prototype.parseText = function (text, characterSet) {
		
		this.text.raw = text;
		this.text.charToIndex = {};
		this.text.characterSet = characterSet ? characterSet : "";
		
		if (characterSet == Silicitect.predefinedCharacterSet) {
			
			this.text.characterSet = "!@#$%^&*()_+{}\":|?><~±§¡€£¢∞œŒ∑´®†¥øØπ∏¬˚∆åÅßΩéúíóáÉÚÍÓÁëüïöäËÜÏÖÄ™‹›ﬁﬂ‡°·—≈çÇ√-=[];',.\\/`µ≤≥„‰◊ˆ˜¯˘¿⁄\n\t";
			this.text.characterSet += "1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
			
		}
		
		for (var a = 0; a < this.text.characterSet.length; a++) {
			
			this.text.charToIndex[this.text.characterSet.charAt(a)] = a;
			
		}
		
		for (var a = 0; a < this.text.raw.length; a++) {
			
			var char = this.text.raw.charAt(a);
			
			if (1 + this.text.charToIndex[char]) continue;
			
			this.text.charToIndex[char] = this.text.characterSet.length;
			this.text.characterSet += char;
			
		}
		
	};
	
	Silicitect.prototype.randomUniform = function () {
		
		this.random.lossless = (4321421413 * this.random.lossless + 432194612 + this.random.seed) % 43214241 * (79143569 + this.random.seed);
		
		return 1e-10 * (this.random.lossless % 1e10);
		
	};
	
	Silicitect.prototype.randomGaussian = function (mean, standardDeviation) {
		
		if (this.random.nextGaussian != -1) {
			
			var output = this.random.nextGaussian;
			
			this.random.nextGaussian = -1;
			
			return output;
			
		}
		
		var xa = 0;
		var xb = 0;
		var w = 0;
		
		do {
			
			xa = 2 * this.randomUniform() - 1;
			xb = 2 * this.randomUniform() - 1;
			w = xa * xa + xb * xb;
			
		} while (w > 1);
		
		w = Math.sqrt((-2 * Math.log(w)) / w);
		
		this.random.nextGaussian = mean + xb * w * standardDeviation;
		
		return mean + xa * w * standardDeviation;
		
	};
	
	Silicitect.prototype.matrix = function (n, d) {
		
		this.memCount += n * d;
		
		return {n:n, d:d, l:n * d, i:this.memCount - n * d};
		
	};
	
	Silicitect.prototype.matrixRandomiseUniform = function (ma) {
		
		for (var a = 0; a < ma.l; a++) this.weights[ma.i + a] = this.randomUniform();
		
		return ma;
		
	};
	
	Silicitect.prototype.matrixRandomiseNormalised = function (ma) {
		
		for (var a = 0; a < ma.l; a++) this.weights[ma.i + a] = this.randomUniform() / Math.sqrt(ma.d);
		
		return ma;
		
	};
	
	Silicitect.prototype.matrixRandomiseGaussian = function (ma, median, standardDeviation) {
		
		for (var a = 0; a < ma.l; a++) this.weights[ma.i + a] = this.randomGaussian(median, standardDeviation);
		
		return ma;
		
	};
	
	Silicitect.prototype.matrixFill = function (ma, valueA) {
		
		for (var a = 0; a < ma.l; a++) this.weights[ma.i + a] = valueA;
		
		return ma;
		
	};
	
	Silicitect.prototype.matrixFillExcept = function (ma, valueA, index, valueB) {
		
		for (var a = 0; a < ma.l; a++) this.weights[ma.i + a] = a == index ? valueB : valueA;
		
		return ma;
		
	};
	
	Silicitect.prototype.matrixScalar = function (ma, scale) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[ma.i + a] *= scale;
			
		}
		
		return out;
		
	};
	
	Silicitect.prototype.matrixSoftmax = function (ma, temp) {
		
		var out = this.matrix(ma.n, ma.d);
		var max = -1e10;
		var sum = 0;
		
		if (temp) {
			
			for (var a = 0; a < ma.l; a++) this.weights[ma.i + a] /= temp;
			
		}
		
		for (var a = 0; a < ma.l; a++) {
			
			if (this.weights[ma.i + a] > max) max = this.weights[ma.i + a];
			
		}
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = Math.exp(this.weights[ma.i + a] - max);
			
			sum += this.weights[out.i + a];
			
		}
		
		for (var a = 0; a < ma.l; a++) this.weights[out.i + a] /= sum;
		
		return out;
		
	};
	
	Silicitect.prototype.matrixSampleMax = function (ma) {
		
		var highest = 0;
		
		for (var a = 1; a < ma.l; a++) {
			
			if (this.weights[ma.i + a] > this.weights[ma.i + highest]) highest = a;
			
		}
		
		return highest;
		
	};
	
	Silicitect.prototype.matrixSampleRandomSum = function (ma) {
		
		var random = this.randomUniform();
		var sum = 0;
		
		for (var a = 0; a < ma.l; a++) {
			
			sum += this.weights[ma.i + a];
			
			if (sum > random) return a;
			
		}
		
		return a - 1;
		
	};
	
	Silicitect.prototype.matrixInvert = function (ma) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = 1 - this.weights[ma.i + a];
			
		}
		
		return out;
		
	};
	
	Silicitect.prototype.matrixClone = function (ma) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = this.weights[ma.i + a];
			
		}
		
		return out;
		
	};
	
	Silicitect.prototype.matrixMultiply = function (ma, mb) {
		
		var out = this.matrix(ma.n, mb.d);
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				this.weights[out.i + mb.d * a + b] = 0;
				
				for (var c = 0; c < ma.d; c++) {
					
					this.weights[out.i + mb.d * a + b] += this.weights[ma.i + ma.d * a + c] * this.weights[mb.i + mb.d * c + b];
					
				}
				
			}
			
		}
		
		if (this.recordBackprop) this.backprop.push(ma, mb, out, Silicitect.matrixMultiplyID);
		
		return out;
		
	};
	
	Silicitect.prototype.matrixMultiplyBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.n; a++) {
			
			for (var b = 0; b < mb.d; b++) {
				
				for (var c = 0; c < ma.d; c++) {
					
					this.dWeights[ma.i + ma.d * a + c] += this.weights[mb.i + mb.d * c + b] * this.dWeights[out.i + mb.d * a + b];
					this.dWeights[mb.i + mb.d * c + b] += this.weights[ma.i + ma.d * a + c] * this.dWeights[out.i + mb.d * a + b];
					
				}
				
			}
			
		}
		
	};
	
	Silicitect.prototype.matrixElementMultiply = function (ma, mb) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = this.weights[ma.i + a] * this.weights[mb.i + a];
			
		}
		
		if (this.recordBackprop) this.backprop.push(ma, mb, out, Silicitect.matrixElementMultiplyID);
		
		return out;
		
	};
	
	Silicitect.prototype.matrixElementMultiplyBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			this.dWeights[ma.i + a] += this.weights[mb.i + a] * this.dWeights[out.i + a];
			this.dWeights[mb.i + a] += this.weights[ma.i + a] * this.dWeights[out.i + a];
			
		}
		
	};
	
	Silicitect.prototype.matrixAdd = function (ma, mb) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = this.weights[ma.i + a] + this.weights[mb.i + a];
			
		}
		
		if (this.recordBackprop) this.backprop.push(ma, mb, out, Silicitect.matrixAddID);
		
		return out;
		
	};
	
	Silicitect.prototype.matrixAddBackward = function (ma, mb, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			this.dWeights[ma.i + a] += this.dWeights[out.i + a];
			this.dWeights[mb.i + a] += this.dWeights[out.i + a];
			
		}
		
	};
	
	Silicitect.prototype.matrixSigmoid = function (ma) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = 1 / (1 + Math.exp(-this.weights[ma.i + a]));
			
		}
		
		if (this.recordBackprop) this.backprop.push(ma, out, Silicitect.matrixSigmoidID);
		
		return out;
		
	};
	
	Silicitect.prototype.matrixSigmoidBackward = function (ma, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			this.dWeights[ma.i + a] += this.weights[out.i + a] * (1 - this.weights[out.i + a]) * this.dWeights[out.i + a];
			
		}
		
	};
	
	Silicitect.prototype.matrixRectifiedLinear = function (ma) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = Math.max(0, this.weights[ma.i + a]);
			
		}
		
		if (this.recordBackprop) this.backprop.push(ma, out, Silicitect.matrixRectifiedLinearID);
		
		return out;
		
	};
	
	Silicitect.prototype.matrixRectifiedLinearBackward = function (ma, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			this.dWeights[ma.i + a] += this.weights[ma.i + a] > 0 ? this.dWeights[out.i + a] : 0;
			
		}
		
	};
	
	Silicitect.prototype.matrixHyperbolicTangent = function (ma) {
		
		var out = this.matrix(ma.n, ma.d);
		
		for (var a = 0; a < ma.l; a++) {
			
			this.weights[out.i + a] = Math.tanh(this.weights[ma.i + a]);
			
		}
		
		if (this.recordBackprop) this.backprop.push(ma, out, Silicitect.matrixHyperbolicTangentID);
		
		return out;
		
	};
	
	Silicitect.prototype.matrixHyperbolicTangentBackward = function (ma, out) {
		
		for (var a = 0; a < ma.l; a++) {
			
			this.dWeights[ma.i + a] += (1 - this.weights[out.i + a] * this.weights[out.i + a]) * this.dWeights[out.i + a];
			
		}
		
	};
	
})();