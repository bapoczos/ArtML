-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target_grams, normalize, target_masks)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target_grams = target_grams
  self.loss = 0
  
  self.target_masks = target_masks
  self.target_masks_means = nil
  self.target_masks_exp = nil

  self.first = true

  self.gram = GramMatrix()
  self.crit = nn.SmoothL1Criterion()

  self.gradInput = nil
end

function StyleLoss:updateOutput(input)
  -- We do everything in updateGradInput to save memory
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  -- Iterate through colors and get gradient
  self.gradInput = self.gradInput or gradOutput:clone()
  self.gradInput:zero()
  self.loss = 0 
  
  -- Expand masks for one time
  if self.first then
    self.first = false
    self.target_masks_exp = {}
    self.target_masks_means = {}

    for k , _ in ipairs(self.target_masks) do
       self.target_masks_exp[k] = self.target_masks[k]:add_dummy():expandAs(input)
       self.target_masks_means[k] = self.target_masks[k]:mean()
       
       -- Delete
       self.target_masks[k] = nil
    end
  end

  -- Apply masks
  for k , _ in ipairs(self.target_masks_exp) do

    -- Forward
    local masked_input = torch.cmul(input,self.target_masks_exp[k])
    local G = self.gram:forward(masked_input)

    if(self.target_masks_means[k] > 0) then
      G:div(input:nElement() * self.target_masks_means[k])
    end

    self.loss = self.loss + self.crit:forward(G, self.target_grams[k])  

    -- Backward
    local dG = self.crit:backward(G, self.target_grams[k])
    if self.target_masks_means[k] > 0 then
      dG:div(input:nElement() * self.target_masks_means[k])
    end

    local gradInput = self.gram:backward(masked_input, dG)
    if self.normalize then
      gradInput:div(torch.norm(gradInput, 1) + 1e-8)
    end
    self.gradInput:add(gradInput)

  end
  self.gradInput:add(gradOutput)
 
  return self.gradInput
end