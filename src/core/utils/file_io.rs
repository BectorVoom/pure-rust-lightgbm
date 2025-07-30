use crate::core::utils::binary_writer::BinaryWriter;
use std::fs;

/// Type alias for size values, compatible with C++ size_t
pub type SizeT = usize;

/// Trait for virtual file writers that can be initialized and support binary writing
/// Trait for virtual file writers that can be initialized and support binary writing
pub trait VirtualFileWriter: BinaryWriter {
    /// Initializes the file writer, returns true on success
    fn init(&mut self) -> bool;
}

/// Trait for virtual file readers that can be initialized and support reading
/// Trait for virtual file readers that can be initialized and support reading
pub trait VirtualFileReader {
    /// Initializes the file reader, returns true on success
    fn init(&mut self) -> bool;

    /// Reads data into the provided buffer, returns the number of bytes read
    fn read(&self, buffer: &mut [u8]) -> usize;
}

/// File I/O operations factory for creating readers and writers
#[derive(Debug)]
pub struct FileIO;

impl FileIO {
    /// Creates a new file writer for the given filename
    pub fn make_writer(filename: &str) -> Box<dyn VirtualFileWriter> {
        todo!("Implementation of concrete file writer")
    }

    /// Creates a new file reader for the given filename
    pub fn make_reader(filename: &str) -> Box<dyn VirtualFileReader> {
        todo!("Implementation of concrete file reader")
    }

    /// Checks if a file exists at the given path
    pub fn exists(filename: &str) -> bool {
        fs::metadata(filename).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::binary_writer::VecBinaryWriter;
    use std::io;

    struct MockFileWriter {
        writer: VecBinaryWriter,
        init_called: bool,
    }

    impl MockFileWriter {
        fn new() -> Self {
            Self {
                writer: VecBinaryWriter::new(),
                init_called: false,
            }
        }
    }

    impl BinaryWriter for MockFileWriter {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.writer.write(data)
        }
    }

    impl VirtualFileWriter for MockFileWriter {
        fn init(&mut self) -> bool {
            self.init_called = true;
            true
        }
    }

    struct MockFileReader {
        data: Vec<u8>,
        init_called: bool,
        read_position: usize,
    }

    impl MockFileReader {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                init_called: false,
                read_position: 0,
            }
        }
    }

    impl VirtualFileReader for MockFileReader {
        fn init(&mut self) -> bool {
            self.init_called = true;
            true
        }

        fn read(&self, buffer: &mut [u8]) -> usize {
            let available = self.data.len().saturating_sub(self.read_position);
            let to_read = buffer.len().min(available);

            if to_read > 0 {
                buffer[..to_read]
                    .copy_from_slice(&self.data[self.read_position..self.read_position + to_read]);
            }

            to_read
        }
    }

    #[test]
    fn test_virtual_file_writer_init() {
        let mut writer = MockFileWriter::new();
        assert!(!writer.init_called);

        let result = writer.init();
        assert!(result);
        assert!(writer.init_called);
    }

    #[test]
    fn test_virtual_file_writer_write() {
        let mut writer = MockFileWriter::new();
        writer.init();

        let data = b"hello world";
        let bytes_written = writer.write(data).unwrap();

        assert_eq!(bytes_written, data.len());
        assert_eq!(writer.writer.buffer(), data);
    }

    #[test]
    fn test_virtual_file_reader_init() {
        let mut reader = MockFileReader::new(vec![1, 2, 3, 4, 5]);
        assert!(!reader.init_called);

        let result = reader.init();
        assert!(result);
        assert!(reader.init_called);
    }

    #[test]
    fn test_virtual_file_reader_read() {
        let mut reader = MockFileReader::new(vec![1, 2, 3, 4, 5]);
        reader.init();

        let mut buffer = [0u8; 3];
        let bytes_read = reader.read(&mut buffer);

        assert_eq!(bytes_read, 3);
        assert_eq!(buffer, [1, 2, 3]);
    }

    #[test]
    fn test_virtual_file_reader_read_partial() {
        let mut reader = MockFileReader::new(vec![1, 2]);
        reader.init();

        let mut buffer = [0u8; 5];
        let bytes_read = reader.read(&mut buffer);

        assert_eq!(bytes_read, 2);
        assert_eq!(&buffer[..2], &[1, 2]);
    }

    #[test]
    fn test_file_io_exists() {
        // Test with a known file (this source file should exist)
        let current_file = file!();
        assert!(FileIO::exists(current_file));

        // Test with a non-existent file
        assert!(!FileIO::exists("/this/file/does/not/exist.txt"));
    }
}
