import React, { useState } from 'react';
import { View, Button, Image, Alert, StyleSheet, Platform, Linking } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const ImageUploadScreen = () => {
  const [imageUri, setImageUri] = useState<string | null>(null);

  // Hàm hiển thị thông báo khi quyền bị từ chối
  const showPermissionAlert = (permissionType: string) => {
    Alert.alert(
      'Quyền cần thiết',
      `Ứng dụng cần quyền truy cập ${permissionType} để tiếp tục sử dụng tính năng này.`,
      [
        { text: 'Hủy', style: 'cancel' },
        {
          text: 'Mở cài đặt',
          onPress: () => Linking.openSettings(),
        },
      ]
    );
  };

  // Hàm yêu cầu quyền
  const requestPermissions = async (): Promise<boolean> => {
    if (Platform.OS === 'ios') {
      const { status: libraryStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();

      if (libraryStatus !== 'granted') {
        showPermissionAlert('thư viện ảnh');
        return false;
      }

      if (cameraStatus !== 'granted') {
        showPermissionAlert('camera');
        return false;
      }

      return true;
    }

    if (Platform.OS === 'android') {
      const { status: libraryStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();

      if (libraryStatus !== 'granted') {
        showPermissionAlert('thư viện ảnh');
        return false;
      }

      if (cameraStatus !== 'granted') {
        showPermissionAlert('camera');
        return false;
      }

      return true;
    }

    return false;
  };

  // Hàm chọn ảnh từ thư viện
  const pickImage = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    } else {
      Alert.alert('Hủy', 'Bạn đã hủy chọn ảnh.');
    }
  };

  // Hàm chụp ảnh từ camera
  const takePhoto = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    } else {
      Alert.alert('Hủy', 'Bạn đã hủy chụp ảnh.');
    }
  };

  // Hàm upload ảnh lên server
  const uploadImage = async () => {
    if (!imageUri) {
      Alert.alert('Chưa có ảnh để upload!');
      return;
    }

    const formData = new FormData();
    const imageFile = {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'photo.jpg',
    } as any;

    formData.append('file', imageFile);

    try {
      const response = await fetch('https://your-server.com/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });

      const responseData = await response.json();
      Alert.alert('Upload thành công!', responseData.message || 'Thành công!');
    } catch (error) {
      console.error(error);
      Alert.alert('Upload thất bại!', 'Vui lòng thử lại.');
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.buttonContainer}>
        <Button title="Chọn ảnh từ thư viện" onPress={pickImage} />
        <Button title="Chụp ảnh mới" onPress={takePhoto} />
      </View>

      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.imagePreview} />
      )}

      <Button title="Upload ảnh" onPress={uploadImage} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
  },
  buttonContainer: {
    marginBottom: 20,
  },
  imagePreview: {
    width: 200,
    height: 200,
    marginVertical: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ddd',
  },
});

export default ImageUploadScreen;
