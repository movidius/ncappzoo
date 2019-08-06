import numpy as numpy
import cv2

class Canvas:

    #IMAGES_ACROSS = 32
    #IMAGES_DOWN = 12
    BOTTOM_INFO_BAR_HEIGHT_MIN = 20
    TOP_INFO_BAR_HEIGHT_MIN = 150

    FPS_TEXT_ROW = 2
    TIMER_TEXT_ROW = 1
    INFERENCE_LABEL_TEXT_ROW = 1
    PAUSE_TEXT_ROW = 1
    LOADING_TEXT_ROW = 1
    DONE_COUNT_TEXT_ROW = 2
    PRESS_ANY_KEY_ROW = 3

    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


    def __init__(self, canvas_width:int, canvas_height:int, images_down:int, images_across:int):

        self._images_down = images_down
        self._images_across = images_across
        self._grid_max_images = self._images_across * self._images_down
        self._grid_max_images_str = str(self._grid_max_images)

        self._text_scale = 1.0
        self._text_background_color = (40, 40, 40)
        self._text_color = (255, 255, 255)  # white text
        text_size = cv2.getTextSize("ZZ", Canvas.TEXT_FONT, self._text_scale, 1)[0]
        self._text_height = text_size[1]
        self._text_bg_height = self._text_height + 14

        #total canvas dimensions
        self._canvas_width = canvas_width
        self._canvas_height = canvas_height

        # for now no use for bottom bar
        self._bottom_bar_height = int(self._canvas_height * 0.01)
        if (self._bottom_bar_height < Canvas.BOTTOM_INFO_BAR_HEIGHT_MIN):
            self._bottom_bar_height = Canvas.BOTTOM_INFO_BAR_HEIGHT_MIN
        self._bottom_bar_width = self._canvas_width

        self._top_bar_height = int(self._canvas_height * 0.1)
        if (self._top_bar_height < Canvas.TOP_INFO_BAR_HEIGHT_MIN):
            self._top_bar_height = Canvas.TOP_INFO_BAR_HEIGHT_MIN
        self._top_bar_width = canvas_width

        # top info bar
        self._top_bar_left = 0
        self._top_bar_right = self._top_bar_left + self._top_bar_width
        self._top_bar_top = 0
        self._top_bar_bottom = self._top_bar_top + self._top_bar_height

        # bottom info bar
        self._bottom_bar_left = 0
        self._bottom_bar_right = self._bottom_bar_left + self._bottom_bar_width
        self._bottom_bar_top = self._canvas_height - self._bottom_bar_height
        self._bottom_bar_bottom = self._bottom_bar_top + self._bottom_bar_height

        #grid dimensions
        self._grid_top = 0 + self._top_bar_height
        max_grid_height = self._canvas_height - self._bottom_bar_height - self._top_bar_height
        max_grid_width = self._canvas_width
        self._grid_line_thickness = 1

        #clear whole canvas to start
        self._canvas_image = numpy.zeros((self._canvas_height, self._canvas_width, 3), numpy.uint8)

        self._image_width = int((max_grid_width-1)/self._images_across)
        self._image_height = int((max_grid_height-1)/self._images_down)

        self._grid_width = self._images_across * self._image_width
        self._grid_left = int((self._canvas_width - self._grid_width)/2)
        self._grid_right = self._grid_left + self._grid_width
        self._grid_height = self._images_down * self._image_height
        self._grid_bottom = self._grid_top + self._grid_height

        self._large_image_width = 112
        self._large_image_height = 112
        self._large_image_left = int(canvas_width/2) - int(self._large_image_width/2)
        self._large_image_right = self._large_image_left + self._large_image_width
        self._large_image_top = 8
        self._large_image_bottom = self._large_image_top + self._large_image_height

        # add some padding for the text that goes on top bar so not right against the edge of window
        self._top_bar_text_left = self._top_bar_left + 10
        self._top_bar_text_top = self._top_bar_top + 10
        self._top_bar_text_left_width = (self._large_image_left - 10) - self._top_bar_text_left


        self._grid_red = 128
        self._grid_green = 128
        self._grid_blue = 128

        self._draw_grid_lines()
        self._done_red = 255
        self._done_green = 255
        self._done_blue = 255

        self._image_done_rect_thickness = 2

        self._grid_image_list = list()
        self._large_image_list = list()

        self._draw_lines_large_to_grid = False

        self._gird_undone_image_transparency = 0.6

        self._num_bar_top_text_rows = 3
        self._top_bar_text_row_tops = [None] * self._num_bar_top_text_rows
        self._top_bar_text_row_tops[0] = 12
        self._top_bar_text_row_tops[1] = self._top_bar_text_row_tops[0] + self._text_bg_height + 10
        self._top_bar_text_row_tops[2] = self._top_bar_text_row_tops[1] + self._text_bg_height + 10

        self._done_count = 0


    def load_images(self, image_list:list):
        self._grid_image_list.clear()
        self._large_image_list.clear()
        transparency = self._gird_undone_image_transparency
        for image_index in range(0, len(image_list)):
            if (image_index >= self._grid_max_images):
                break
            temp_large_image = cv2.resize(image_list[image_index], (self._large_image_width, self._large_image_height))
            self._large_image_list.append(temp_large_image)
            temp_image = cv2.resize(image_list[image_index], (self._image_width, self._image_height))
            self._grid_image_list.append(temp_image)
            self._draw_grid_image(image_index, transparency)
        return


    def reset_canvas(self):

        #clear whole canvas to start
        self._canvas_image = numpy.zeros((self._canvas_height, self._canvas_width, 3), numpy.uint8)
        self._draw_grid_lines()
        self._draw_undone_images()
        self._done_count = 0


    def _draw_undone_images(self):
        for image_index in range(0, len(self._grid_image_list)):
            if (image_index >= self._grid_max_images):
                break
            self._draw_grid_image(image_index, self._gird_undone_image_transparency)


    def _draw_grid_image(self, image_index:int, transparency:float):
        if (image_index >= self._grid_max_images):
            return
        image_left, image_top, image_right, image_bottom = self._get_image_square(image_index)
        self._canvas_image[image_top:image_bottom, image_left:image_right] = \
            cv2.addWeighted(self._canvas_image[image_top:image_bottom, image_left:image_right], transparency,
                            self._grid_image_list[image_index], 1.0 - transparency, 0.0)


    def _draw_large_image(self, image_index:int, transparency:float):
        if (image_index >= self._grid_max_images):
            return

        #image_left, image_top, image_right, image_bottom = self._get_image_square(image_index)
        self._canvas_image[self._large_image_top:self._large_image_bottom, self._large_image_left:self._large_image_right] = \
            cv2.addWeighted(self._canvas_image[self._large_image_top:self._large_image_bottom, self._large_image_left:self._large_image_right], transparency,
                            self._large_image_list[image_index], 1.0 - transparency, 0.0)


    def set_draw_lines(self, val:bool):
        self._draw_lines_large_to_grid = val


    def _draw_grid_lines(self):
        blue = self._grid_blue
        green = self._grid_green
        red = self._grid_red

        # lines going across
        for line_index in range(0, self._images_down+1):
            line_y = self._grid_top + (line_index * self._image_height)
            cv2.line(self._canvas_image, (self._grid_left, line_y), (self._grid_right, line_y), (blue, green, red),
                     self._grid_line_thickness)

        #lines going down
        for line_index in range(0, self._images_across+1):
            line_x = self._grid_left + (line_index * self._image_width)
            cv2.line(self._canvas_image, (line_x, self._grid_top), (line_x, self._grid_top + ((self._images_down) * self._image_height)), (blue, green, red),
                     self._grid_line_thickness)



    def mark_image_done(self, image_index:int, label_text:str=None):
        self._done_count += 1
        if (image_index >= self._grid_max_images):
            return
        self._draw_grid_image(image_index, 0.0)
        self._draw_large_image(image_index, 0.0)
        image_left, image_top, image_right, image_bottom = self._get_image_square(image_index)
        cv2.rectangle(self._canvas_image, (image_left, image_top), (image_right, image_bottom),
                      (self._done_blue, self._done_green, self._done_red), self._image_done_rect_thickness)

        if (label_text != None):
            self.draw_inference_label(label_text)

        self.draw_done_count()

        if (self._draw_lines_large_to_grid) :
            cv2.line(self._canvas_image,
                     (image_left+int(self._image_width/2), image_top + int(self._image_height/2)),
                     (self._large_image_left + int(self._large_image_width/2), self._large_image_bottom), (255, 0, 0),  1)


    def _get_image_square(self, image_index:int):
        row = int(image_index / self._images_across)
        col = image_index - (row * self._images_across)
        image_left = self._grid_left + (self._image_width * col)
        image_top = self._grid_top + (self._image_height * row)
        image_right = image_left + self._image_width
        image_bottom = image_top + self._image_height
        return image_left, image_top, image_right, image_bottom


    def get_canvas_image(self):
        return self._canvas_image


    def show_loading(self):
        self._put_text_top_bar_left("Loading Images...", Canvas.LOADING_TEXT_ROW)


    def clear_loading(self):
        left, top, right, bottom = self._get_top_bar_left_text_bg_rect(Canvas.LOADING_TEXT_ROW)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)

    def pause_start(self):
        self._put_text_top_bar_left("Paused...", Canvas.PAUSE_TEXT_ROW)
        

    def press_any_key(self):
        self._put_text_top_bar_left("Press any key to continue...", Canvas.PRESS_ANY_KEY_ROW)
        
    def press_quit_key(self):
        self._put_text_top_bar_left("Press q to quit.", Canvas.PRESS_ANY_KEY_ROW)

    def show_device(self, device:str):
        self._put_text_top_bar_right("Device: "+ device, Canvas.PRESS_ANY_KEY_ROW)
        
    def clear_press_any_key(self):
        left, top, right, bottom = self._get_top_bar_left_text_bg_rect(Canvas.PRESS_ANY_KEY_ROW)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)

    def pause_stop(self):
        left, top, right, bottom = self._get_top_bar_left_text_bg_rect(Canvas.PAUSE_TEXT_ROW)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)


    def draw_inference_label(self, label_text:str):
        self._put_text_top_bar_left(label_text, Canvas.INFERENCE_LABEL_TEXT_ROW)

    def draw_done_count(self):
        draw_str = "Images: " + str(self._done_count) +"/" + self._grid_max_images_str
        self._put_text_top_bar_left(draw_str, Canvas.DONE_COUNT_TEXT_ROW)

    def hide_done_count(self):
        left, top, right, bottom = self._get_top_bar_left_text_bg_rect(Canvas.DONE_COUNT_TEXT_ROW)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)

    def clear_top_bar(self):
        clear_image = numpy.full((self._top_bar_height, self._top_bar_width, 3),
                                 (0, 0, 0), numpy.uint8)
        self._canvas_image[self._top_bar_top:self._top_bar_bottom, self._top_bar_left: self._top_bar_right] = clear_image


    def show_fps(self, fps:float):
        fps_str = "FPS: %2.1f" % fps
        self._put_text_top_bar_right(fps_str, Canvas.FPS_TEXT_ROW)

    def hide_fps(self):
        left, top, right, bottom = self._get_top_bar_right_text_bg_rect(Canvas.FPS_TEXT_ROW)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)


    def show_timer(self, time:float):
        time_str = "Elapsed: %3.1f" % time
        self._put_text_top_bar_right(time_str, Canvas.TIMER_TEXT_ROW)

    def hide_timer(self):
        left, top, right, bottom = self._get_top_bar_right_text_bg_rect(Canvas.TIMER_TEXT_ROW)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)

    def _put_text_top_bar_right(self, text:str, text_row:int=1):
        left, top, right, bottom = self._get_top_bar_right_text_bg_rect(text_row)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)
        text_top_index = text_row -1
        self._put_text_on_canvas(text, -1, self._top_bar_text_row_tops[text_top_index], 0)

    def _put_text_top_bar_left(self, text:str, text_row:int=1):
        left, top, right, bottom = self._get_top_bar_left_text_bg_rect(text_row)
        cv2.rectangle(self._canvas_image, (left, top), (right, bottom),
                      self._text_background_color, -1)
        text_top_index = text_row -1
        self._put_text_on_canvas(text, self._top_bar_text_left, self._top_bar_text_row_tops[text_top_index], 0)


    def _get_top_bar_right_text_leftmost(self):
        return self._large_image_right + 10

    def _get_top_bar_left_text_leftmost(self):
        return self._top_bar_text_left

    def _get_top_bar_right_text_bg_rect(self, text_row:int):
        left = self._get_top_bar_right_text_leftmost()
        text_top_index = text_row - 1
        top = self._top_bar_text_row_tops[text_top_index] - 4
        right = self._canvas_width - 10
        bottom = top + self._text_bg_height
        return (left, top, right, bottom)

    def _get_top_bar_left_text_bg_rect(self, text_row:int):
        left = self._get_top_bar_left_text_leftmost()
        text_top_index = text_row - 1
        top = self._top_bar_text_row_tops[text_top_index] - 4
        right = self._top_bar_text_left + self._top_bar_text_left_width
        bottom = top + self._text_bg_height
        return (left, top, right, bottom)

    def _put_text_on_canvas(self, text:str, text_left:int, text_top: int, text_min_width:int):

        text_size = cv2.getTextSize(text, Canvas.TEXT_FONT, self._text_scale, 1)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        if (text_left == -1):
            display_image_width = self._canvas_image.shape[1]
            text_left = display_image_width - text_width - 10

        text_bottom = text_top + text_height

        cv2.putText(self._canvas_image, text, (text_left, text_bottom), cv2.FONT_HERSHEY_SIMPLEX, self._text_scale,
                    self._text_color, 1)
