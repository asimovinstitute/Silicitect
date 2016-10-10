// generic parameters
var running = false;
var iterationsPerFrame = 100;
var maxIterations = 2000;
var averageBatchTime = 0;
var totalIterations = 0;
var sil = null;
var layerSizes = [];

Stecy.setup = function () {
	
	Art.title = "Silicitect AE demo";
	
	Art.width = 500;
	Art.height = 100;
	Art.useCanvas = true;
	Art.stretch = 2;
	
};

Art.ready = function () {
	
	init();
	
	Art.doStyle(0, "whiteSpace", "pre-wrap", "font", "20px monospace", "tabSize", "4", "background", "#333", "color", "#ccc");
	
};

function init (e) {
	
	sil = new Silicitect();
	
	layerSizes = [5, 4, 3, 4, 5];
	
	sil.reguliser = 1e-8;
	sil.learningRate = 0.1;
	sil.clipValue = 5;
	sil.decay = 0.95;
	sil.decayLinear = 0.9;
	sil.optimiser = Silicitect.adamOptimiser;
	
	sil.init(initAE);
	
	running = true;
	
}

function doNetworkStuff () {
	
	if (!running || totalIterations >= maxIterations) return;
	else totalIterations += iterationsPerFrame;
	
	sil.startLearningSession();
	
	for (var b = 0; b < iterationsPerFrame; b++) {
		
		trainAutoencoder();
		
	}
	
	sil.endLearningSession();
	
	averageBatchTime += sil.batchTime;
	
	Art.doWrite(0, totalIterations + " " + (sil.totalLoss / iterationsPerFrame).toFixed(2) + " " + sil.batchTime + "ms");
	Art.doWrite(0, " avg " + (averageBatchTime / (totalIterations / iterationsPerFrame)).toFixed(0));
	
	// drawAutoencoder();
	printAutoencoder();
	
}

Stecy.sequence("update", [doNetworkStuff]);

function trainVariationalAutoencoder () {
	
	
	
}

function printVariationalAutoencoder () {
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		sil.weights[sil.net["input"].i + a] = +(sil.randomUniform() < 0.5);
		
	}
	
	updateVAE();
	
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
	
	
	updateAE();
	
	sil.computeLoss("output", "input", "matrixClone", Silicitect.linearLoss);
	sil.backpropagate();
	
}

function printAutoencoder () {
	
	for (var a = 0; a < sil.net["input"].l; a++) {
		
		sil.weights[sil.net["input"].i + a] = +(sil.randomUniform() < 0.5);
		
	}
	
	updateAE();
	
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
	
	updateAE();
	
}

function initVAE () {
	
	var net = sil.net;
	
	net["input"] = sil.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		if (a == (layerSizes.length - 1) / 2) {
			
			net["mean"] = sil.matrixRandomiseNormalised(sil.matrix(size, prevSize));
			net["meanBias"] = sil.matrixFill(sil.matrix(size, 1), 1);
			
			net["standardDeviation"] = sil.matrixRandomiseNormalised(sil.matrix(size, prevSize));
			net["standardDeviationBias"] = sil.matrixFill(sil.matrix(size, 1), 1);
			
		} else {
			
			net["hidden" + a] = sil.matrixRandomiseNormalised(sil.matrix(size, prevSize));
			net["hiddenBias" + a] = sil.matrixFill(sil.matrix(size, 1), 1);
			
			net["outputLast" + a] = sil.matrix(size, 1);
			
		}
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = sil.matrixRandomiseNormalised(sil.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = sil.matrixFill(sil.matrix(lastLayerSize, 1), 1);
	
	net["output"] = sil.matrix(lastLayerSize, 1);
	
}

function updateVAE () {
	
	var net = sil.net;
	
	
	
}

function initAE () {
	
	var net = sil.net;
	
	net["input"] = sil.matrix(layerSizes[0], 1);
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var prevSize = layerSizes[a - 1];
		var size = layerSizes[a];
		
		net["hidden" + a] = sil.matrixRandomiseNormalised(sil.matrix(size, prevSize));
		net["hiddenBias" + a] = sil.matrixFill(sil.matrix(size, 1), 1);
		
		net["outputLast" + a] = sil.matrix(size, 1);
		
	}
	
	var lastLayerSize = layerSizes[layerSizes.length - 1];
	
	net["decoder"] = sil.matrixRandomiseNormalised(sil.matrix(lastLayerSize, layerSizes[layerSizes.length - 2]));
	net["decoderBias"] = sil.matrixFill(sil.matrix(lastLayerSize, 1), 1);
	
	net["output"] = sil.matrix(lastLayerSize, 1);
	
}

function updateAE () {
	
	var net = sil.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["input"] : net["outputLast" + (a - 1)];
		
		var h0 = sil.matrixMultiply(net["hidden" + a], previousLayer);
		
		net["outputLast" + a] = sil.matrixSigmoid(sil.matrixAdd(h0, net["hiddenBias" + a]));
		
	}
	
	net["output"] = sil.matrixSigmoid(sil.matrixAdd(sil.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]));
	
}