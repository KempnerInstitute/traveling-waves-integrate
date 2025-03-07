# Traveling Waves Integrate Spatial Information Through Time

Code for the preprint "Traveling Waves Integrate Spatial Information Through Time".

**Abstract**: Traveling waves of neural activity are widely observed in the brain, but their precise computational function remains unclear. One prominent hypothesis is that they enable the transfer and integration of spatial information across neural populations. However, few computational models have explored how traveling waves might be harnessed to perform such integrative processing. Drawing inspiration from the famous "Can one hear the shape of a drum?" problem -- which highlights how normal modes of wave dynamics encode geometric information -- we investigate whether similar principles can be leveraged in artificial neural networks. Specifically, we introduce convolutional recurrent neural networks that learn to produce traveling waves in their hidden states in response to visual stimuli, enabling spatial integration. By then treating these wave-like activation sequences as visual representations themselves, we obtain a powerful representational space that outperforms local feed-forward networks on tasks requiring global spatial context. In particular, we observe that traveling waves effectively expand the receptive field of locally connected neurons, supporting long-range encoding and communication of information. We demonstrate that models equipped with this mechanism solve visual semantic segmentation tasks demanding global integration, significantly outperforming local feed-forward models and rivaling non-local U-Net models with fewer parameters. As a first step toward traveling-wave-based communication and visual representation in artificial networks, our findings suggest wave-dynamics may provide efficiency and training stability benefits, while simultaneously offering a new framework for connecting models to biological recordings of neural activity.

Correspondence to: Mozes Jacobs (mozesjacobs@g.harvard.edu) and Andy Keller (takeller@fas.harvard.edu).


## Polygons NWM Waves
<table>
  <tr>
    <td><img src="gifs/polygons1.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/polygons2.gif" alt="Description 2" width="200"/></td>
  </tr>
</table>

## Multi-MNIST NWM Waves
<table>
  <tr>
    <td><img src="gifs/multi_mnist_cornn_sample-1.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/multi_mnist_cornn_sample-2.gif" alt="Description 2" width="200"/></td>
    <td><img src="gifs/multi_mnist_cornn_sample-3.gif" alt="Description 3" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/multi_mnist_cornn_sample-4.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/multi_mnist_cornn_sample-5.gif" alt="Description 2" width="200"/></td>
    <td><img src="gifs/multi_mnist_cornn_sample-6.gif" alt="Description 3" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/multi_mnist_cornn_sample-7.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/multi_mnist_cornn_sample-8.gif" alt="Description 2" width="200"/></td>
    <td><img src="gifs/multi_mnist_cornn_sample-9.gif" alt="Description 3" width="200"/></td>
  </tr>
</table>

## Tetrominoes NWM Waves
<table>
  <tr>
    <td><img src="gifs/tetronimoes_cornn_sample-1.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_cornn_sample-2.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_cornn_sample-3.gif" alt="Description 1" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/tetronimoes_cornn_sample-4.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_cornn_sample-5.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_cornn_sample-6.gif" alt="Description 1" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/tetronimoes_cornn_sample-7.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_cornn_sample-8.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_cornn_sample-9.gif" alt="Description 1" width="200"/></td>
  </tr>
</table>

## Tetrominoes LSTM Waves
<table>
  <tr>
    <td><img src="gifs/tetronimoes_lstm_sample-1.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_lstm_sample-2.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_lstm_sample-3.gif" alt="Description 1" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/tetronimoes_lstm_sample-4.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_lstm_sample-5.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_lstm_sample-6.gif" alt="Description 1" width="200"/></td>
  </tr>
  <tr>
    <td><img src="gifs/tetronimoes_lstm_sample-7.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_lstm_sample-8.gif" alt="Description 1" width="200"/></td>
    <td><img src="gifs/tetronimoes_lstm_sample-9.gif" alt="Description 1" width="200"/></td>
  </tr>
</table>

## Instructions
1. Set up the conda environment using environment.yml.
2. Create data by running data_scripts/create_tetrominoes.py and data_scripts/download_mnist.py data_scripts/create_multi_mnist.py
3. Edit dataset_config.py with absolute paths to the generated data.
4. Train all the models by 'cd' into scripts-ccn-1, then 'cd' into each folder and run the desired training scripts.
5. Produce MNIST and Tetrominoes scores using produce_scores.py
6. Produce Multi-MNIST scores using produce_scores_multi-mnist.py
7. Process the MNIST and Tetrominoes scores using process_scores_df.ipynb and process_scores_df_min_max.ipynb
8. Process Multi-MNIST scores using process_scores_df_multi-mnist.ipynb
9. Produce the LSTM and NWM figure (Figure 5) using produce_fig_lstm_nwm.ipynb
10. Produce the square area theory results using Square_Area_Theory.ipynb
11. Produce the multi polygon results using Multi_Polygon_Classification.ipynb