I want to model various sources.

1. I have information about the source. 72 measurements with freq, spl
(and potentially phase) taken at 0, 10, 20 ..., up 350 degrees
horizontally and taken at 0, 10, 20 up to 350 degrees vertically.
There may also be measurements every 5 degrees instead of 10. datas
are available either in 2 files (horizontal, vertical) in a csv format
OR with 72 files with angle and _H for horizontal and _V vertical in
the name.

2. I want to model the source by interpolating the source linearly
between the known values and define the pressure field on a sphere 
around the source. since it is a sphere interpolation is on the sphere
not via a straight line between the point.

3. make this model available in room_acoustics

-------------------------------------------------------------------------

I want to have more tests to check the accuracy of the system.
We already have comparison between analytical solutions in 1d, 2d, and
3d.
1. Check that the relative error is decreasing when the size of the grid
and the number of iteration increases.
2. Check that it works for a dense grid (without partionning) and then
with partionning. Check that it give the same results with and without
accelerate.

-------------------------------------------------------------------------

Now we want to fix all tests 1 by 1.
Then we want to remove most warnings from cargo check
cargo test --release need to be clean
