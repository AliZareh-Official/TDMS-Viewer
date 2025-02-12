import numpy as np
import pygame
from scipy.ndimage import gaussian_filter1d
from nptdms import TdmsFile

# Configuration
DATA_PATH = r'C:\Users\Ali\Desktop\Odor Project\Data\Simul\tdms\neural_data.tdms'
WIN_WIDTH, WIN_HEIGHT = 1500, 800
COLORS = [
    (0, 119, 190), (213, 94, 0), (0, 158, 115),
    (204, 121, 167), (230, 159, 0)
]
SAMPLING_RATE_HZ = 30000  # Sampling frequency in Hz
TAO_INCREMENT_SECONDS = 0.1  # Time window increment for zooming


class TDMSVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("TDMS Data Visualizer - Professional")

        # Initialize core components first
        self.raw_data = self._load_data()
        self._precompute_filters()

        # Initialize data state before layout calculations
        self.current_ch = 0
        self.tao_seconds = 0.1  # Initial time window in seconds
        self.tao_samples = int(self.tao_seconds * SAMPLING_RATE_HZ)  # Convert to samples
        self.offset = 0  # Current offset in samples
        self.zoom_level = 1.0
        self.multi_view = True  # Multi-channel view by default
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.y_scale = 1.0

        # Configure layout after state initialization
        self.layout = {
            'sidebar': {
                'width': 220,
                'margin': 20,
                'color': (28, 33, 39)
            },
            'plot': {
                'header_height': 40,
                'footer_height': 30,
                'channel_spacing': 15,
                'bg_color': (255, 255, 255),
                'grid_color': (235, 235, 235)
            },
            'button': {
                'height': 40,
                'spacing': 10,
                'color': (44, 49, 58),
                'hover': (64, 69, 78),
                'active': (84, 89, 98),
                'radius': 6
            }
        }
        self._calculate_layout()

        # Initialize UI components last
        self.font = pygame.font.Font(None, 18)
        self.buttons = self._create_buttons()
        self.tooltips = {
            'zoom_in': "Zoom In (Mouse Wheel Up/+)",
            'zoom_out': "Zoom Out (Mouse Wheel Down/-)",
            'prev': "Pan Left (←)",
            'next': "Pan Right (→)",
            'tao_up': "Increase Window (↑)",
            'tao_down': "Decrease Window (↓)",
            'multi': "Toggle Multi-View (M)",
            'prev_channel': "Previous Channel (PgUp)",  # New tooltip
            'next_channel': "Next Channel (PgDown)"    # New tooltip
        }
        self.running = True

    def _load_data(self):
        """Loads TDMS file and extracts channel data."""
        try:
            tdms_file = TdmsFile.read(DATA_PATH)
            group = tdms_file.groups()[0]
            channels = group.channels()
            return {
                'names': [ch.name for ch in channels],
                'data': [ch[:] for ch in channels],
                'length': len(channels[0][:]) if channels else 0
            }
        except Exception as e:
            print(f"Error loading TDMS file: {e}")
            return {'names': [], 'data': [], 'length': 0}

    def _precompute_filters(self):
        """Precomputes filtered versions of the data for efficient rendering."""
        self.filtered_cache = []
        max_tao = self.raw_data['length']
        for ch_data in self.raw_data['data']:
            pyramid = []
            current_data = ch_data.copy()
            while len(current_data) > 10:
                pyramid.append({
                    'data': current_data,
                    'sigma': max(1, len(current_data) / max_tao * 10)
                })
                current_data = current_data[::2]  # Downsample by half
            self.filtered_cache.append(pyramid)

    def _create_buttons(self):
        """Creates UI buttons for user interaction."""
        sb = self.layout['sidebar']
        btn = self.layout['button']
        buttons = []
        y = sb['margin']
        for name, label in [
            ('zoom_in', 'Zoom In (+)'),
            ('zoom_out', 'Zoom Out (-)'),
            ('prev', '← Previous'),
            ('next', 'Next →'),
            ('tao_up', 'Tao ↑'),
            ('tao_down', 'Tao ↓'),
            ('multi', 'Multi View (M)'),
            ('prev_channel', 'Prev Channel (PgUp)'),  # New button
            ('next_channel', 'Next Channel (PgDown)') # New button
        ]:
            rect = pygame.Rect(
                sb['margin'], y,
                sb['width'] - 2 * sb['margin'],
                btn['height']
            )
            buttons.append((name, label, rect))
            y += btn['height'] + btn['spacing']
        return buttons

    def _calculate_layout(self):
        """Dynamically calculates layout dimensions based on window size."""
        sb = self.layout['sidebar']
        pl = self.layout['plot']

        # Sidebar dimensions
        sb['rect'] = pygame.Rect(0, 0, sb['width'], WIN_HEIGHT)

        # Plot area dimensions
        plot_x = sb['width'] + sb['margin']
        plot_width = WIN_WIDTH - plot_x - sb['margin']
        plot_height = WIN_HEIGHT - pl['header_height'] - pl['footer_height']
        pl['rect'] = pygame.Rect(
            plot_x,
            pl['header_height'],
            plot_width,
            plot_height
        )

        # Channel dimensions
        num_channels = len(self.raw_data['data']) if self.multi_view else 1
        self.channel_height = (plot_height -
                                (num_channels - 1) * pl['channel_spacing']) / num_channels

    def _get_visible_data(self, channel):
        """Retrieves visible data for a specific channel based on zoom and offset."""
        pyramid = self.filtered_cache[channel]
        if not pyramid:
            return None
        level = min(int(np.log2(max(1, self.tao_samples // 100))), len(pyramid) - 1)
        level_data = pyramid[level]
        data_len = len(level_data['data'])
        window_size = min(data_len, max(1, self.tao_samples // (2 ** level)))
        start = max(0, min(self.offset // (2 ** level), data_len - window_size))
        end = start + window_size
        if start >= end:
            start = max(0, data_len - window_size)
            end = data_len
        return {
            'x': np.arange(start, end) * (2 ** level),
            'y': gaussian_filter1d(level_data['data'][start:end], sigma=level_data['sigma']),
            'step': 2 ** level
        }

    def _draw_interface(self):
        """Draws the entire interface, including sidebar, plot area, and buttons."""
        self.screen.fill(self.layout['plot']['bg_color'])

        # Draw sidebar
        sb = self.layout['sidebar']
        pygame.draw.rect(self.screen, sb['color'], sb['rect'])

        # Draw plot area
        pl = self.layout['plot']
        pygame.draw.rect(self.screen, (240, 240, 240), pl['rect'], border_radius=8)

        # Draw components
        self._draw_header_footer()
        self._draw_buttons()
        self._draw_channels()

    def _draw_header_footer(self):
        """Draws the header and footer of the plot area."""
        pl = self.layout['plot']
        header_rect = pygame.Rect(pl['rect'].x, 0, pl['rect'].width, pl['header_height'])
        footer_rect = pygame.Rect(pl['rect'].x, WIN_HEIGHT - pl['footer_height'], pl['rect'].width, pl['footer_height'])

        # Header
        pygame.draw.rect(self.screen, (250, 250, 250), header_rect)
        title = self.font.render("TDMS Signal Viewer", True, (40, 40, 40))
        self.screen.blit(title, (header_rect.x + 15, header_rect.centery - 10))

        # Footer
        pygame.draw.rect(self.screen, (245, 245, 245), footer_rect)
        time_info = f"Window: {self.offset / SAMPLING_RATE_HZ:.2f}s - {(self.offset + self.tao_samples) / SAMPLING_RATE_HZ:.2f}s"
        footer_text = self.font.render(time_info, True, (100, 100, 100))
        self.screen.blit(footer_text, (footer_rect.x + 15, footer_rect.centery - 10))

    def _draw_buttons(self):
        """Draws interactive buttons on the sidebar."""
        mouse_pos = pygame.mouse.get_pos()
        btn = self.layout['button']
        for name, label, rect in self.buttons:
            color = btn['color']
            if rect.collidepoint(mouse_pos):
                color = btn['hover']
            if (name == 'multi' and self.multi_view) or (name == 'tao_up' and self.tao_seconds >= 5):
                color = btn['active']
            pygame.draw.rect(self.screen, color, rect, border_radius=btn['radius'])
            text = self.font.render(label, True, (240, 240, 240))
            self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_channels(self):
        """Draws all visible channels in the plot area."""
        pl = self.layout['plot']
        channels = range(len(self.raw_data['data'])) if self.multi_view else [self.current_ch]
        for i, ch in enumerate(channels):
            y_pos = pl['rect'].y + i * (self.channel_height + pl['channel_spacing'])
            chan_rect = pygame.Rect(
                pl['rect'].x + 10,
                y_pos + 5,
                pl['rect'].width - 20,
                self.channel_height - 10
            )

            # Channel background
            pygame.draw.rect(self.screen, (248, 248, 248), chan_rect, border_radius=6)

            # Channel data
            self._draw_channel_data(ch, chan_rect)

            # Channel label
            label = self.font.render(self.raw_data['names'][ch], True, COLORS[ch % len(COLORS)])
            self.screen.blit(label, (chan_rect.x + 10, chan_rect.y + 5))

    def _draw_channel_data(self, channel, rect):
        """Draws the signal data for a specific channel."""
        vis_data = self._get_visible_data(channel)
        if not vis_data or len(vis_data['y']) == 0:
            return
        y_min, y_max = vis_data['y'].min(), vis_data['y'].max()
        y_range = y_max - y_min if y_max != y_min else 1
        x_min, x_max = vis_data['x'][0], vis_data['x'][-1]

        scaled_y = (vis_data['y'] - y_min) / y_range * self.y_scale
        points = [
            (
                rect.x + (x - x_min) / (x_max - x_min) * rect.width,
                rect.y + rect.height - ((y - y_min) / y_range * self.y_scale) * rect.height
            )
            for x, y in zip(vis_data['x'], vis_data['y'])
        ]
        if len(points) > 1:
            pygame.draw.lines(self.screen, COLORS[channel % len(COLORS)], False, points, 2)

    def _handle_input(self):
        """Handles user input events."""
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                global WIN_WIDTH, WIN_HEIGHT
                WIN_WIDTH, WIN_HEIGHT = event.size
                self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), pygame.RESIZABLE)
                self._calculate_layout()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse click
                    self.dragging = True
                    self.last_mouse_pos = mouse_pos
                    # Check for button click
                    for name, label, rect in self.buttons:
                        if rect.collidepoint(mouse_pos):
                            self._handle_button_click(name)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                self.offset = max(0, min(
                    self.raw_data['length'] - self.tao_samples,
                    self.offset - int(dx * self.tao_samples / self.layout['plot']['rect'].width)
                ))
                self.last_mouse_pos = mouse_pos
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_zoom_in() if event.y > 0 else self._handle_zoom_out()
            elif event.type == pygame.KEYDOWN:
                key_handlers = {
                    pygame.K_LEFT: self._handle_prev,
                    pygame.K_RIGHT: self._handle_next,
                    pygame.K_PLUS: self._handle_zoom_in,
                    pygame.K_EQUALS: self._handle_zoom_in,
                    pygame.K_MINUS: self._handle_zoom_out,
                    pygame.K_UP: self._handle_tao_up,
                    pygame.K_DOWN: self._handle_tao_down,
                    pygame.K_m: self._handle_multi,
                    pygame.K_w: self._handle_y_scale_up,  # Bind 'W' to increase vertical scaling
                    pygame.K_s: self._handle_y_scale_down  # Bind 'S' to decrease vertical scaling
                }
                if event.key in key_handlers:
                    key_handlers[event.key]()

    def _handle_prev_channel(self):
        """Switches to the previous channel in single-channel mode."""
        if not self.multi_view:
            self.current_ch = max(0, self.current_ch - 1)
            self._calculate_layout()  # Recalculate layout to reflect the new channel

    def _handle_next_channel(self):
        """Switches to the next channel in single-channel mode."""
        if not self.multi_view:
            self.current_ch = min(len(self.raw_data['data']) - 1, self.current_ch + 1)
            self._calculate_layout()  # Recalculate layout to reflect the new channel

    def _handle_button_click(self, button_name):
        """Handles button click events."""
        handlers = {
            'zoom_in': self._handle_zoom_in,
            'zoom_out': self._handle_zoom_out,
            'prev': self._handle_prev,
            'next': self._handle_next,
            'tao_up': self._handle_tao_up,
            'tao_down': self._handle_tao_down,
            'multi': self._handle_multi,
            'prev_channel': self._handle_prev_channel,  # New handler
            'next_channel': self._handle_next_channel   # New handler
        }
        if button_name in handlers:
            handlers[button_name]()

    def _handle_zoom_in(self):
        """Zooms in by reducing the time window."""
        self.zoom_level *= 1.2
        self.tao_seconds = max(0.01, self.tao_seconds / 1.2)
        self.tao_samples = int(self.tao_seconds * SAMPLING_RATE_HZ)

    def _handle_zoom_out(self):
        """Zooms out by increasing the time window."""
        self.zoom_level *= 0.8
        self.tao_seconds = min(self.raw_data['length'] / SAMPLING_RATE_HZ, self.tao_seconds * 1.2)
        self.tao_samples = int(self.tao_seconds * SAMPLING_RATE_HZ)

    def _handle_prev(self):
        """Pans to the previous section of the data."""
        self.offset = max(0, self.offset - self.tao_samples // 2)

    def _handle_next(self):
        """Pans to the next section of the data."""
        self.offset = min(self.raw_data['length'] - self.tao_samples,
                          self.offset + self.tao_samples // 2)

    def _handle_tao_up(self):
        """Increases the time window."""
        self.tao_seconds = min(10.0, self.tao_seconds + TAO_INCREMENT_SECONDS)
        self.tao_samples = int(self.tao_seconds * SAMPLING_RATE_HZ)

    def _handle_tao_down(self):
        """Decreases the time window."""
        self.tao_seconds = max(0.1, self.tao_seconds - TAO_INCREMENT_SECONDS)
        self.tao_samples = int(self.tao_seconds * SAMPLING_RATE_HZ)

    def _handle_multi(self):
        """Toggles between single-channel and multi-channel views."""
        self.multi_view = not self.multi_view
        self._calculate_layout()

    def _handle_y_scale_up(self):
        self.y_scale *= 1.2

    def _handle_y_scale_down(self):
        self.y_scale = max(1, self.y_scale / 1.2)

    def run(self):
        """Main loop to handle visualization."""
        clock = pygame.time.Clock()
        while self.running:
            self._handle_input()
            self._draw_interface()
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    visualizer = TDMSVisualizer()
    visualizer.run()