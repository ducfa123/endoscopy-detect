import { useState } from 'react';
import { Image, StyleSheet, TouchableOpacity, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

const API_URL = 'http://194.163.137.133:8000/call_service';
const BASE_URL = 'http://194.163.137.133:8000';

export default function UploadScreen() {
  const [image, setImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      alert('Xin lỗi, chúng tôi cần quyền truy cập thư viện ảnh!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    
    if (status !== 'granted') {
      alert('Xin lỗi, chúng tôi cần quyền truy cập camera!');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

const handleUpload = async () => {
  if (!image) {
    alert('Vui lòng chọn hoặc chụp một ảnh để tải lên!');
    return;
  }

  try {
    setIsLoading(true);
    const formData = new FormData();
    
    const timestamp = new Date().getTime();
    const filename = `image_${timestamp}.jpg`;

    formData.append('image', {
      uri: image,
      type: 'image/jpeg',
      name: filename
    } as any);

    const uploadResponse = await fetch(API_URL, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
      },
    });

    const result = await uploadResponse.json();

    if (result.status === 'success') {
      // Tạo params object trước để kiểm tra data
      const params = {
        originalImage: result.response.img_path || '',
        imgPath: result.response.img_path || '',
        endoscopyImages: JSON.stringify(result.response.endoscopy_img_list_path || []),
        lesionImages: JSON.stringify(result.response.lesion_list_path || []),
        logId: result.response?.logId || '',
        sessionId: result.response?.sessionId || ''
      };

      // Log để debug
      console.log('Response data:', result.response);
      console.log('Prepared navigation params:', params);

      // Navigation trực tiếp, không dùng setTimeout
      router.push({
        pathname: "/result",
        params
      });
    } else {
      throw new Error(result.message || 'Tải lên thất bại');
    }
  } catch (error: any) {
    console.error('Upload error:', {
      name: error.name,
      message: error.message,
      stack: error.stack
    });
    alert(`Có lỗi xảy ra khi tải ảnh lên: ${error.message}`);
  } finally {
    setIsLoading(false);
  }
};

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title" style={styles.title}>
        Tải ảnh lên
      </ThemedText>

      <View style={styles.imageSection}>
        <TouchableOpacity
          style={styles.imagePicker}
          onPress={pickImage}
          disabled={isLoading}
        >
          {image ? (
            <Image source={{ uri: image }} style={styles.preview} />
          ) : (
            <ThemedText>Chọn ảnh từ thư viện</ThemedText>
          )}
        </TouchableOpacity>

        <View style={styles.buttonGroup}>
          <TouchableOpacity
            style={[styles.actionButton, styles.galleryButton, isLoading && styles.buttonDisabled]}
            onPress={pickImage}
            disabled={isLoading}
          >
            <Ionicons name="images" size={24} color="white" />
            <ThemedText type="default" style={styles.buttonText}>
              Thư viện ảnh
            </ThemedText>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.cameraButton, isLoading && styles.buttonDisabled]}
            onPress={takePhoto}
            disabled={isLoading}
          >
            <Ionicons name="camera" size={24} color="white" />
            <ThemedText type="default" style={styles.buttonText}>
              Chụp ảnh
            </ThemedText>
          </TouchableOpacity>
        </View>
      </View>

      <TouchableOpacity
        style={[styles.uploadButton, (!image || isLoading) && styles.buttonDisabled]}
        onPress={handleUpload}
        disabled={!image || isLoading}
      >
        <ThemedText type="default" style={{ color: 'white' }}>
          {isLoading ? 'Đang tải lên...' : 'Tải lên'}
        </ThemedText>
      </TouchableOpacity>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
  },
  imageSection: {
    width: '100%',
    alignItems: 'center',
    marginBottom: 20,
  },
  imagePicker: {
    width: 300,
    height: 300,
    borderWidth: 2,
    borderColor: '#ddd',
    borderStyle: 'dashed',
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  preview: {
    width: '100%',
    height: '100%',
    borderRadius: 15,
  },
  buttonGroup: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 20,
  },
  actionButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 5,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  galleryButton: {
    backgroundColor: '#34C759',
  },
  cameraButton: {
    backgroundColor: '#5856D6',
  },
  buttonText: {
    marginLeft: 8,
    color: 'white',
  },
  uploadButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#cccccc',
  },
});