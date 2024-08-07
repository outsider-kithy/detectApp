import React, { useEffect, useRef, useState } from 'react';
import { SafeAreaView, View, Text, Image, StyleSheet } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { fetch, decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as mobilenet from '@tensorflow-models/mobilenet';

const App = () => {
  const [isTfReady, setIsTfReady] = useState(false);
  const [result, setResult] = useState('');
  const image = useRef(null);

  const load = async () => {
    try {
      // Load mobilenet.
      await tf.ready();
      const model = await mobilenet.load();
      setIsTfReady(true);

      // Start inference and show result.
      const image = require('./assets/myImage.jpg');
      const imageAssetPath = Image.resolveAssetSource(image);
      const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
      const imageDataArrayBuffer = await response.arrayBuffer();
      const imageData = new Uint8Array(imageDataArrayBuffer);
      const imageTensor = decodeJpeg(imageData);
      const prediction = await model.classify(imageTensor);
      if (prediction && prediction.length > 0) {
        let predictionArray = [];
        for(let i = 0; i < prediction.length; i++){
          console.log(prediction[i]);
          predictionArray.push(prediction[i].className);
        }

        setResult(
          `${predictionArray.join(", ")}`
        );
      }
    } catch (err) {
      console.log(err);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.box}>
        <Image
          ref={image}
          source={require('./assets/myImage.jpg')}
          style={styles.image}
        />
      </View>
      {!isTfReady && <Text>Loading TFJS model...</Text>}
      {isTfReady && result === '' && <Text>Classifying...</Text>}
      {result !== '' && <Text style={styles.result}>{result}</Text>}
     </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  box:{
    position: 'relative',
    width: 300,
    height: 300,
  },
  image:{
    position: 'absolute',
    top: 0,
    width: 300,
    height: 300,
  },
  result: {
    marginTop: 20,
    fontWeight: 'bold',
    fontSize: 20,
    color: 'red',
  }
});

export default App;

