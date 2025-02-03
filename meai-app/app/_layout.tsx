import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import 'react-native-reanimated';

import { useColorScheme } from '@/hooks/useColorScheme';

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const [loaded] = useFonts({
    SpaceMono: require('../assets/fonts/SpaceMono-Regular.ttf'),
  });

  useEffect(() => {
    if (loaded) {
      SplashScreen.hideAsync();
    }
  }, [loaded]);

  if (!loaded) {
    return null;
  }

  return (
    <>
      <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
        <Stack>
          <Stack.Screen 
            name="login" 
            options={{ 
              title: 'Đăng nhập',
              headerShown: false,  // Ẩn header ở màn hình login
              headerBackVisible: false,
              gestureEnabled: false 
            }} 
          />
          <Stack.Screen 
            name="upload" 
            options={{ 
              title: 'Tải ảnh lên',
              // Không cho phép quay lại màn hình login
              headerBackVisible: false,
              gestureEnabled: false
            }} 
          />
          <Stack.Screen 
            name="result" 
            options={{ 
              title: 'Kết quả phân tích',
              headerBackVisible: true,
              gestureEnabled: true
            }} 
          />
          <Stack.Screen name="+not-found" options={{ title: 'Oops!' }} />
        </Stack>
      </ThemeProvider>
      <StatusBar style="auto" />
    </>
  );
}