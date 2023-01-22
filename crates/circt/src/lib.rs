use std::mem;

use itertools::Itertools;

use circt_sys as ffi;
use mlir::{
    attr::{
        ArrayAttr, DictionaryAttr, FlatSymbolRefAttr, FUNCTION_ARG_DICT_ATTR_NAME,
        FUNCTION_RESULT_DICT_ATTR_NAME,
    },
    Attribute, Identifier, Location, NamedAttribute, Operation, OperationState, StringRef,
    SymbolTable, Type,
};

/// Defines an extension trait on a type with trait methods of the form `fn(&self) -> bool`.
macro_rules! is_fns_ext {
    (
        $v:vis trait $tr:ident: $name:ident$(<$lt:lifetime>)? {
            $(fn $fn_name:ident = $ffi_name:path;)*
        }
    ) => {
        $v trait $tr {
            $(fn $fn_name(&self) -> bool;)*
        }

        impl $(<$lt>)? $tr for $name $(<$lt>)? {
            $(fn $fn_name(&self) -> bool {
                unsafe { $ffi_name(self.as_raw().into()) }
            })*
        }
    };
}
pub(crate) use is_fns_ext;

pub mod hw {
    use mlir::{
        attr::{TypeAttr, FUNCTION_TYPE_ATTR_NAME},
        ty::FunctionType,
        Block, Region,
    };

    use super::*;

    use self::ty::TypeExt;

    pub mod ty {
        use super::*;

        is_fns_ext! {
            pub trait TypeExt: Type {
                fn is_hw_array_type = ffi::hwTypeIsAArrayType;
                fn is_hw_inout = ffi::hwTypeIsAInOut;
                fn is_hw_int_type = ffi::hwTypeIsAIntType;
                fn is_hw_struct_type = ffi::hwTypeIsAStructType;
                fn is_hw_type_alias_type = ffi::hwTypeIsATypeAliasType;
                fn is_hw_value_type = ffi::hwTypeIsAValueType;
            }
        }

        #[repr(C)]
        pub struct StructFieldInfo {
            pub name: Identifier,
            pub ty: Type,
        }

        const _: () =
            assert!(mem::size_of::<StructFieldInfo>() == mem::size_of::<ffi::HWStructFieldInfo>());

        pub fn struct_ty(elements: &[StructFieldInfo]) -> Option<Type> {
            mlir::context().without_mutex(|cx| {
                unsafe {
                    let raw = ffi::hwStructTypeGet(
                        cx.into(),
                        elements.len() as isize,
                        // SAFETY: StructFieldInfo is #[repr(C)] and its fields are transparent wrappers
                        // around their underlying types.
                        elements.as_ptr() as *const ffi::HWStructFieldInfo,
                    );
                    Type::from_raw(raw.into())
                }
            })
        }
    }

    /// A module port direction.
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub enum PortDirection {
        Input = 1,
        Output = 2,
        InOut = 3,
    }

    /// The name, type and direction of a single port.
    #[derive(Clone)]
    pub struct PortInfo {
        pub name: Identifier,
        pub direction: PortDirection,
        pub ty: Type,
        pub sym: Option<Identifier>,
    }

    /// The ports of a module.
    #[derive(Default)]
    pub struct ModulePortInfo {
        pub inputs: Vec<PortInfo>,
        pub outputs: Vec<PortInfo>,
    }

    pub struct ModuleOpInfo<'a> {
        pub name: StringRef<'a>,
        pub ports: ModulePortInfo,
        pub params: ArrayAttr,
        pub comment: Option<Identifier>,
        pub attrs: &'a [NamedAttribute],
    }

    fn export_port(sym: Identifier) -> NamedAttribute {
        NamedAttribute::get("hw.exportPort", FlatSymbolRefAttr::new(sym.value()))
    }

    /// Wraps inout parameters in `InOutType` and returns as a `Type`.
    fn wrap_inout(port: &PortInfo) -> Type {
        if port.direction == PortDirection::InOut && !port.ty.is_hw_inout() {
            todo!("wrap ty in InOutType")
        } else {
            port.ty
        }
    }

    /// Generate the attributes dict for a port.
    fn port_attrs(port: &PortInfo) -> Attribute {
        DictionaryAttr::create(
            port.sym
                .map(export_port)
                .as_ref()
                .map(std::slice::from_ref)
                .unwrap_or_default(),
        )
        .into()
    }

    pub struct ModuleOp {
        op: Operation,
    }

    impl ModuleOp {
        // See `buildModule()` in HWOps.cpp
        pub fn build(info: ModuleOpInfo<'_>) -> ModuleOp {
            let mut state = OperationState::get("hw.module", Location::unknown());
            state.add_attribute(SymbolTable::symbol_attribute_name(), info.name);

            let (arg_names, arg_types, arg_attrs) = info
                .ports
                .inputs
                .iter()
                .map(|port| {
                    (
                        Attribute::from(port.name),
                        wrap_inout(port),
                        port_attrs(port),
                    )
                })
                .multiunzip::<(Vec<Attribute>, Vec<Type>, Vec<Attribute>)>();

            let (result_names, result_types, result_attrs) = info
                .ports
                .outputs
                .iter()
                .map(|port| (Attribute::from(port.name), port.ty, port_attrs(port)))
                .multiunzip::<(Vec<Attribute>, Vec<Type>, Vec<Attribute>)>();

            let ty = FunctionType::get(&arg_types, &result_types);

            state.add_attribute(FUNCTION_TYPE_ATTR_NAME, TypeAttr::get(ty.into()));
            state.add_attribute("argNames", ArrayAttr::create(&arg_names));
            state.add_attribute("resultNames", ArrayAttr::create(&result_names));
            state.add_attribute(FUNCTION_ARG_DICT_ATTR_NAME, ArrayAttr::create(&arg_attrs));
            state.add_attribute(
                FUNCTION_RESULT_DICT_ATTR_NAME,
                ArrayAttr::create(&result_attrs),
            );
            state.add_attribute("parameters", info.params);
            state.add_attribute(
                "comment",
                info.comment.unwrap_or_else(|| Identifier::get("")),
            );
            state.add_attributes(info.attrs);

            let mut region = state.add_region(Region::create());
            let mut block = region.append_block(Block::create());

            for input in &info.ports.inputs {
                block.add_argument(input.ty, Location::unknown());
            }

            todo!()
        }
    }
}

mod private {
    pub trait Sealed {}
}
