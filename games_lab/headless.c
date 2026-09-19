/* Headless graphics boundary only. Game mechanics remain upstream code. */
#include "raylib.h"
bool IsWindowReady(void) { return false; }
bool IsKeyDown(int key) { (void)key; return false; }
bool IsKeyPressed(int key) { (void)key; return false; }
void InitWindow(int w, int h, const char* title) { (void)w; (void)h; (void)title; }
void CloseWindow(void) {}
void SetTargetFPS(int fps) { (void)fps; }
void BeginDrawing(void) {}
void EndDrawing(void) {}
void ClearBackground(Color c) { (void)c; }
void DrawRectangle(int x, int y, int w, int h, Color c) { (void)x; (void)y; (void)w; (void)h; (void)c; }
void DrawText(const char* text, int x, int y, int n, Color c) { (void)text; (void)x; (void)y; (void)n; (void)c; }
void DrawRectangleRounded(Rectangle r, float a, int b, Color c) { (void)r; (void)a; (void)b; (void)c; }
void DrawRectangleRoundedLinesEx(Rectangle r, float a, int b, float d, Color c) { (void)r; (void)a; (void)b; (void)d; (void)c; }
int MeasureText(const char* text, int size) { (void)text; (void)size; return 0; }
int GetScreenWidth(void) { return 0; }
int GetScreenHeight(void) { return 0; }
