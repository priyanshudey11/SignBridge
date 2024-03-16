import React, { useState,  useEffect} from 'react';
import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

export default function App() {
  const [hasCamreaPermission, setHasCameraPermission] = useState(null);
  const [image, setImage] =  useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [flash, setFlash] = useState(Camera.Constants.Flashmode.off);
  const cameraRef = useRef(null);

  return (
    <View style={styles.container}>
      <Text>An app that translates sign languages into English!</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
