// generic parameters
var running = false;
var iterationsPerFrame = 100;
var maxIterations = 2000;
var averageBatchTime = 0;
var totalIterations = 0;
var sil = null;
var layerSizes = [];

Stecy.setup = function () {
	
	Art.title = "Silicitect FF demo";
	
};

Art.ready = function () {
	
	init();
	
	Art.doStyle(0, "whiteSpace", "pre-wrap", "font", "20px monospace", "tabSize", "4", "background", "#333", "color", "#ccc");
	
};

function init (e) {
	
	sil = new Silicitect();
	
	layerSizes = [2, 2, 1];
	
	sil.reguliser = 1e-8;
	sil.learningRate = 0.1;
	sil.clipValue = 5;
	sil.decay = 0.95;
	sil.decayLinear = 0.9;
	sil.optimiser = Silicitect.adamOptimiser;
	
	sil.init(initFF);
	
	running = true;
	
}

function doNetworkStuff () {
	
	if (!running || totalIterations >= maxIterations) return;
	else totalIterations += iterationsPerFrame;
	
	sil.startLearningSession();
	
	for (var b = 0; b < iterationsPerFrame; b++) {
		
		trainLogicGate();
		
	}
	
	sil.endLearningSession();
	
	averageBatchTime += sil.batchTime;
	
	Art.doWrite(0, totalIterations + " " + (sil.totalLoss / iterationsPerFrame).toFixed(2) + " " + sil.batchTime + "ms");
	Art.doWrite(0, " avg " + (averageBatchTime / (totalIterations / iterationsPerFrame)).toFixed(0));
	
	printLogicGate();
	
}

Stecy.sequence("update", [doNetworkStuff]);

function trainLogicGate () {
	
	sil.weights[sil.net["input"].i + 0] = +(sil.randomUniform() > 0.5);
	sil.weights[sil.net["input"].i + 1] = +(sil.randomUniform() > 0.5);
	sil.weights[sil.net["desiredValues"].i] = +(sil.weights[sil.net["input"].i + 0] + sil.weights[sil.net["input"].i + 1] > 0);
	
	updateFF();
	
	sil.computeLoss("output", "desiredValues", "matrixClone", Silicitect.linearLoss);
	sil.backpropagate();
	
}

function printLogicGate () {
	
	sil.weights[sil.net["input"].i + 0] = 0;
	sil.weights[sil.net["input"].i + 1] = 0;
	updateFF();
	Art.doWrite(0, "\n00 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	
	sil.weights[sil.net["input"].i + 0] = 0;
	sil.weights[sil.net["input"].i + 1] = 1;
	updateFF();
	Art.doWrite(0, "01 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	
	sil.weights[sil.net["input"].i + 0] = 1;
	sil.weights[sil.net["input"].i + 1] = 0;
	updateFF();
	Art.doWrite(0, "10 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	
	sil.weights[sil.net["input"].i + 0] = 1;
	sil.weights[sil.net["input"].i + 1] = 1;
	updateFF();
	Art.doWrite(0, "11 " + sil.weights[sil.net["output"].i].toFixed(2) + "\n");
	
}

function initFF () {
	
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
	net["desiredValues"] = sil.matrix(lastLayerSize, 1);
	
}

function updateFF () {
	
	var net = sil.net;
	
	for (var a = 1; a < layerSizes.length - 1; a++) {
		
		var previousLayer = a == 1 ? net["input"] : net["outputLast" + (a - 1)];
		
		var h0 = sil.matrixMultiply(net["hidden" + a], previousLayer);
		
		net["outputLast" + a] = sil.matrixSigmoid(sil.matrixAdd(h0, net["hiddenBias" + a]));
		
	}
	
	net["output"] = sil.matrixSigmoid(sil.matrixAdd(sil.matrixMultiply(net["decoder"], net["outputLast" + (layerSizes.length - 2)]), net["decoderBias"]));
	
}