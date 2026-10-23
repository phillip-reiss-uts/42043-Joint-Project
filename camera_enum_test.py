#/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -c "
import pyrealsense2 as rs
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
p.start(c); print('started OK')
p.stop()
