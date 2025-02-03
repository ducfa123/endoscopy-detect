import React, { useState } from 'react';
import { Image, ScrollView, StyleSheet, View, ActivityIndicator } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

const BASE_URL = 'http://194.163.137.133:8000';

export default function ResultScreen() {
  const [loadingStates, setLoadingStates] = useState<{[key: string]: boolean}>({});
  const params = useLocalSearchParams();
  
  // Parse các mảng ảnh từ string JSON
  let endoscopyImageList: string[] = [];
  let lesionImageList: string[] = [];
  
  try {
    endoscopyImageList = params.endoscopyImages ? JSON.parse(params.endoscopyImages as string) : [];
    lesionImageList = params.lesionImages ? JSON.parse(params.lesionImages as string) : [];
  } catch (error) {
    console.error('Error parsing image arrays:', error);
    return (
      <ThemedView style={styles.container}>
        <ThemedText type="title">Có lỗi xảy ra khi tải dữ liệu</ThemedText>
      </ThemedView>
    );
  }

  const handleLoadStart = (imageId: string) => {
    setLoadingStates(prev => ({...prev, [imageId]: true}));
  };

  const handleLoadEnd = (imageId: string) => {
    setLoadingStates(prev => ({...prev, [imageId]: false}));
  };

  const getFullUrl = (path: string) => {
    if (path.startsWith('http')) return path;
    return `${BASE_URL}${path}`;
  };

  const renderImageSection = (title: string, images: string[]) => (
    <View style={styles.section}>
      <ThemedText type="title" style={styles.sectionTitle}>
        {title}
      </ThemedText>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        {images.map((imagePath, index) => {
          const imageId = `${title}-${index}`;
          return (
            <View key={imageId} style={styles.imageContainer}>
              <Image
                source={{ uri: getFullUrl(imagePath) }}
                style={styles.image}
                resizeMode="cover"
                onLoadStart={() => handleLoadStart(imageId)}
                onLoadEnd={() => handleLoadEnd(imageId)}
              />
              {loadingStates[imageId] && (
                <ActivityIndicator
                  style={styles.loader}
                  size="large"
                  color="#007AFF"
                />
              )}
              <ThemedText type="default" style={styles.imageText}>
                {`Ảnh ${index + 1}`}
              </ThemedText>
            </View>
          );
        })}
      </ScrollView>
    </View>
  );

  const originalImageId = 'original';

  return (
    <ThemedView style={styles.container}>
      <ScrollView>
        {/* Hiển thị ảnh gốc */}
        <View style={styles.section}>
          <ThemedText type="title" style={styles.sectionTitle}>
            Ảnh gốc
          </ThemedText>
          <View style={styles.imageContainer}>
            <Image
              source={{ uri: getFullUrl(params.originalImage as string) }}
              style={styles.originalImage}
              resizeMode="contain"
              onLoadStart={() => handleLoadStart(originalImageId)}
              onLoadEnd={() => handleLoadEnd(originalImageId)}
            />
            {loadingStates[originalImageId] && (
              <ActivityIndicator
                style={styles.loader}
                size="large"
                color="#007AFF"
              />
            )}
          </View>
        </View>

        {/* Hiển thị danh sách ảnh nội soi */}
        {endoscopyImageList.length > 0 && 
          renderImageSection('Kết quả nội soi', endoscopyImageList)}

        {/* Hiển thị danh sách ảnh tổn thương */}
        {lesionImageList.length > 0 && 
          renderImageSection('Phát hiện tổn thương', lesionImageList)}
      </ScrollView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 10,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    paddingHorizontal: 10,
  },
  originalImage: {
    width: '100%',
    height: 300,
    borderRadius: 10,
  },
  imageContainer: {
    marginRight: 10,
    alignItems: 'center',
    position: 'relative',
  },
  image: {
    width: 200,
    height: 200,
    borderRadius: 10,
  },
  imageText: {
    marginTop: 5,
    fontSize: 12,
  },
  loader: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: [{ translateX: -12 }, { translateY: -12 }],
  }
});