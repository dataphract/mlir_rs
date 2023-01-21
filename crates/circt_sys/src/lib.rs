#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

macro_rules! mlir_type_conversions {
    ($($name:ident),* $(,)?) => {
        $(
            impl From<mlir_sys::$name> for crate::$name {
                fn from(value: mlir_sys::$name) -> crate::$name {
                    crate::$name { ptr: value.ptr }
                }
            }

            impl From<crate::$name> for mlir_sys::$name {
                fn from(value: crate::$name) -> mlir_sys::$name {
                    mlir_sys::$name { ptr: value.ptr }
                }
            }
        )*
    };
}

mlir_type_conversions! {
    MlirContext,
    MlirType,
}
