# Theme Toggle Button Implementation

## Overview
Added a theme toggle button to the Course Materials Assistant interface, positioned in the top-right corner of the header. The button allows users to switch between dark and light themes with smooth animations.

## Features Implemented

### 1. Toggle Button Design
- **Icon-based design**: Uses sun and moon SVG icons to represent light and dark themes
- **Size**: 44px circular button with 22px border-radius
- **Position**: Top-right corner of the header
- **Styling**: Matches existing design aesthetic with consistent colors and borders

### 2. Smooth Animations
- **Icon transitions**: 0.3s cubic-bezier transition for smooth icon rotation and opacity changes
- **Button interactions**: Scale effects on hover (1.05x) and active state (0.95x)
- **Icon rotation**: Sun icon rotates -180° when switching to light theme, moon icon rotates 180° when switching to dark theme
- **Opacity transitions**: Icons fade in/out smoothly when toggling

### 3. Theme Implementation
- **Dark theme (default)**: Original color scheme preserved
- **Light theme**: New color variables with light background, dark text, and adjusted contrast
- **Local storage**: Theme preference saved and restored on page reload
- **System preference detection**: Respects user's OS theme preference as initial setting

### 4. Accessibility Features
- **ARIA label**: Dynamic label that updates based on current theme ("Switch to light/dark theme")
- **Keyboard navigation**: Full keyboard support with Enter and Space key activation
- **Focus indicators**: Visual focus ring matching the existing design system
- **High contrast**: Both themes maintain good color contrast ratios

### 5. Responsive Design
- **Mobile layout**: Button repositions appropriately on smaller screens
- **Touch targets**: 44px minimum touch target size for mobile accessibility
- **Flexible positioning**: Adapts to header layout changes on different screen sizes

## Files Modified

### `/frontend/index.html`
- Added theme toggle button with sun/moon SVG icons
- Wrapped title and subtitle in `.header-content` div for proper layout
- Updated header structure to accommodate the toggle button

### `/frontend/style.css`
- Made header visible and updated layout (was previously hidden)
- Added light theme CSS variables
- Implemented theme toggle button styles with smooth transitions
- Added responsive design updates for mobile devices
- Enhanced icon animation states for both themes

### `/frontend/script.js`
- Added theme toggle functionality
- Implemented local storage for theme persistence
- Added system preference detection
- Enhanced accessibility with dynamic ARIA labels
- Added keyboard event handling for toggle button

## Technical Details

### Theme Toggle Logic
1. **Initialization**: Checks localStorage for saved preference, falls back to system preference
2. **Toggle mechanism**: Adds/removes `.light-theme` class on document body
3. **Persistence**: Saves theme choice to localStorage
4. **Icon states**: CSS controls icon visibility and animation based on body class

### CSS Transitions
- **Timing function**: `cubic-bezier(0.4, 0, 0.2, 1)` for smooth, natural animations
- **Duration**: 0.3s for optimal perceived performance
- **Properties animated**: opacity, transform (rotation and scale), background-color, border-color

### Color Accessibility
- **Dark theme**: High contrast white text on dark backgrounds
- **Light theme**: Dark text on light backgrounds with sufficient contrast ratios
- **Interactive states**: Clear hover and focus indicators in both themes

## Usage
- Click the toggle button to switch between light and dark themes
- Use keyboard (Enter or Space) to activate the toggle
- Theme preference is automatically saved and restored on page reload
- Button shows appropriate icon (sun for dark theme, moon for light theme)