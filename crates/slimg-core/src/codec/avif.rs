use crate::error::{Error, Result};
use crate::format::Format;
use super::{Codec, EncodeOptions, ImageData};

/// AVIF codec (stashed due to build issues on Windows)
pub struct AvifCodec;

impl Codec for AvifCodec {
    fn format(&self) -> Format {
        Format::Avif
    }

    fn decode(&self, _data: &[u8]) -> Result<ImageData> {
        Err(Error::Decode("AVIF decoding is disabled in this build".to_string()))
    }

    fn encode(&self, _image: &ImageData, _options: &EncodeOptions) -> Result<Vec<u8>> {
        Err(Error::Encode("AVIF encoding is disabled in this build".to_string()))
    }
}
