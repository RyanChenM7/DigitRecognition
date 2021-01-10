# DigitRecognition

Standard practice in creating a Neural Network, following multiple online guides on how to implement the classic ANN for beginners in Python.
Includes Matplotlib visualization, which shows the neural network in action. Graphs the output layer for random samples from the MNIST database; this displays the prediction confidence of the network for the image.

The file `neural_network.py` is used to train the neural network using the MNIST database based on a couple of variables.
For instance the number of layers, size of layers, # of epochs, learning rate, and mini batch size.

The file `network_demo.py` will load up a previously trained (or possibly untrained) neural network, and it will run the AI on samples that are pulled from the database.

</br>

Here are some demo examples, loaded from a network trained for 30 minutes on my laptop (about 96% accuracy).

Very confident in 2.
<img src="https://media.discordapp.net/attachments/795803904075366400/797638112418201680/unknown.png?width=1239&height=677"
     alt=""
     style="float: left; margin-right: 10px;" />

Very confident in 4.
<img src="https://media.discordapp.net/attachments/795803904075366400/797626663180566618/unknown.png?width=1239&height=677"
     alt=""
     style="float: left; margin-right: 10px;" />
     
Uncertain, but still got the correct answer of 0.
<img src="https://media.discordapp.net/attachments/795803904075366400/797637160965505034/unknown.png?width=1239&height=677"
     alt=""
     style="float: left; margin-right: 10px;" />

Got the answer wrong, but to be fair it does look like a 5.
<img src="https://media.discordapp.net/attachments/795803904075366400/797636534337011762/unknown.png?width=1239&height=677"
     alt=""
     style="float: left; margin-right: 10px;" />

Was uncertain and guessed 2, but it was actually a weirdly written 7.
<img src="https://media.discordapp.net/attachments/795803904075366400/797636380938207282/unknown.png?width=1239&height=677"
     alt=""
     style="float: left; margin-right: 10px;" /> 
