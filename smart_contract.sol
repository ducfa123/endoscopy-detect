// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Contract {
    address public seller;
    address public buyer;
    uint256 public amount;

    // Constructor để khởi tạo hợp đồng với địa chỉ của người bán
    constructor(address _seller) {
        seller = _seller;
    }

    // Hàm Deposit: cho phép người mua gửi tiền vào hợp đồng
    function deposit() external payable {
        // Hợp đồng có thể chấp nhận thanh toán, và số tiền sẽ bị khóa trong hợp đồng
    }

    // Hàm Release: giải phóng số tiền cho người bán (khi điều kiện được thỏa mãn)
    function release() external {
        require(msg.sender == buyer, "Chỉ người mua mới có thể giải phóng tiền");
        payable(seller).transfer(address(this).balance);  // Chuyển toàn bộ số tiền trong hợp đồng cho người bán
    }

    // Hàm để lấy địa chỉ người mua
    function getBuyer() external view returns (address) {
        return buyer;
    }

    // Hàm để lấy địa chỉ người bán
    function getSeller() external view returns (address) {
        return seller;
    }

    // Hàm để lấy số tiền hiện tại trong hợp đồng
    function getAmount() external view returns (uint256) {
        return amount;
    }

    // Hàm để thiết lập địa chỉ người mua (chỉ có thể gọi bởi người bán)
    function setBuyer(address _buyer) external {
        require(msg.sender == seller, "Chỉ người bán mới có thể thiết lập người mua");
        buyer = _buyer;
    }
}
