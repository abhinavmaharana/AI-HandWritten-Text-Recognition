from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance

from Dataloader import Dataloader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

class FilePaths:
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'


def train(model, loader):
    # number of training epochs since start
    epoch = 0
    # valication character error rate
    bestCharErrorRate = float('inf')
    # number of epochs no improvement of character error rate occured
    noImprovementSince = 0
    # stop training after this no. of epochs
    earlyStopping = 5
    while True:
        epoch +=1
        
        print('Epoch:', epoch)

        #training
        print('Train Neural Network')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/',iterInfo[1],'Loss:',loss)

        #validating
        charErrorRate = validate(model,loader)

        #if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1 
        
        #stopping training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break

def validate(model, loader):
    "Validate Neural Network"
    print('Validate Neural Network')
    loader.validationSet()

    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0 
    numWordTotal = 0

    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0 
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    
    #printing validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%, Word accuracy: %f%%.' % (charErrorRate*100.0,wordAccuracy*100.0))
    return charErrorRate

def infer(model, fnImg):
    "recognize text in the image provided by the file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])

def main():
    "main function"
    #optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the neural network' , action='store_true')
    parser.add_argument('--validate', help='validate the neural network', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--dump', help='dump output of the neural network to csv file', action='store-true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    #training or validating on IAM dataset
    if args.train or args.validate:
        # load training data, create tensorflow model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # saving characters of the model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        #saving words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # Executing training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

        # infering text on test image
        else:
            print(open(FilePaths.fnAccuracy).read())
            model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
            infer(model, FilePaths.fnInfer)

if __name__ == '__main__':
    main()
